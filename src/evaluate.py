import json
import math
from pathlib import Path

try:
    from .retrieve import retrieve
except ImportError:
    from retrieve import retrieve


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "evaluation_questions.json"
K = 5


def get_source_path(item):
    """Read a source path from a retrieval result."""
    return (
        item.get("path")
        or item.get("source")
        or item.get("metadata", {}).get("source")
        or ""
    ).replace("\\", "/").lower()


def matches_relevant_doc(result_path, relevant_doc):
    """Match a labelled path even if the result includes a parent folder."""
    result_path = result_path.strip("/")
    relevant_doc = relevant_doc.replace("\\", "/").lower().strip("/")

    return (
        result_path == relevant_doc
        or result_path.endswith("/" + relevant_doc)
    )


def recall_at_k(results, relevant_docs, k=K):
    """Fraction of labelled relevant documents retrieved in top k."""
    relevant_docs = set(relevant_docs)

    if not relevant_docs:
        return 0.0

    found = set()

    for item in results[:k]:
        path = get_source_path(item)

        for doc in relevant_docs:
            if matches_relevant_doc(path, doc):
                found.add(doc)

    return len(found) / len(relevant_docs)


def reciprocal_rank(results, relevant_docs):
    """Reciprocal rank of the first relevant retrieved document."""
    for rank, item in enumerate(results, start=1):
        path = get_source_path(item)

        if any(
            matches_relevant_doc(path, doc)
            for doc in relevant_docs
        ):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(results, relevant_docs, k=K):
    """Binary-relevance Normalized Discounted Cumulative Gain."""
    relevant_docs = set(relevant_docs)

    if not relevant_docs:
        return 0.0

    gains = []

    for item in results[:k]:
        path = get_source_path(item)

        relevant = any(
            matches_relevant_doc(path, doc)
            for doc in relevant_docs
        )
        gains.append(1.0 if relevant else 0.0)

    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )

    ideal_relevant_count = min(len(relevant_docs), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    print("\nAGRICULTURE RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for index, case in enumerate(dataset, start=1):
        question = case["question"].strip()
        relevant_docs = case["relevant_docs"]

        if not question or not relevant_docs:
            print(f"Skipping invalid test case {index}")
            continue

        # Uses the existing BM25 + dense + RRF + reranking pipeline.
        results = retrieve(question)

        recall = recall_at_k(results, relevant_docs)
        mrr = reciprocal_rank(results, relevant_docs)
        ndcg = ndcg_at_k(results, relevant_docs)

        recall_scores.append(recall)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        print(f"\nTest {index}: {question}")
        print(f"Retrieved documents: {len(results)}")
        print(f"Recall@{K}: {recall:.4f}")
        print(f"Reciprocal Rank: {mrr:.4f}")
        print(f"nDCG@{K}: {ndcg:.4f}")

        print("Ranked source documents:")
        for rank, item in enumerate(results[:K], start=1):
            print(f"  {rank}. {get_source_path(item) or 'Unknown source'}")

    count = len(recall_scores)

    if count == 0:
        raise ValueError("No valid evaluation cases were processed.")

    print("\n" + "=" * 60)
    print(f"SUMMARY — {count} TEST QUESTIONS")
    print("=" * 60)
    print(f"Mean Recall@{K}: {sum(recall_scores) / count:.4f}")
    print(f"Mean MRR:        {sum(mrr_scores) / count:.4f}")
    print(f"Mean nDCG@{K}:   {sum(ndcg_scores) / count:.4f}")
    print("\nScores range from 0 to 1; higher is better.")


if __name__ == "__main__":
    main()
