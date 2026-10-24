
import json
import math
from pathlib import Path

from .retrieve import (
    collection,
    embedding_model,
    bm25_index,
    indexed_records,
    tokenize,
    BGE_QUERY_PREFIX,
    retrieve,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_questions.json"
K = 5


# --------------------------------------------------
# Source helpers
# --------------------------------------------------

def normalize_path(path):
    """Normalize document paths for matching."""
    return str(path or "").replace("\\", "/").lower().lstrip("./")


def get_source(item):
    """Extract a source path from a retrieval result."""
    metadata = item.get("metadata") or {}

    path = (
        item.get("source")
        or metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("filename")
        or metadata.get("path")
        or ""
    )

    return normalize_path(path)


def is_relevant(result, relevant_docs):
    """Check whether a result matches a labelled relevant document."""
    result_path = get_source(result)

    if not result_path:
        return False

    for relevant_doc in relevant_docs:
        label = normalize_path(relevant_doc)

        if (
            result_path == label
            or result_path.endswith("/" + label)
            or label.endswith("/" + result_path)
        ):
            return True

    return False


def deduplicate_by_source(results):
    """Keep the highest-ranked result for each source document."""
    unique_results = []
    seen_sources = set()

    for result in results:
        source_path = get_source(result)

        # Retain results without a source, but do not merge them together.
        if not source_path:
            unique_results.append(result)
            continue

        if source_path not in seen_sources:
            seen_sources.add(source_path)
            unique_results.append(result)

    return unique_results


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def calculate_metrics(results, relevant_docs, k=K):
    """
    Calculate document-level Recall@K, MRR, and nDCG@K.

    Results are deduplicated by source before ranking metrics
    are calculated. Relevance is binary at the document level.
    """
    relevant_docs = list(dict.fromkeys(
        normalize_path(doc)
        for doc in relevant_docs
        if normalize_path(doc)
    ))

    if not relevant_docs:
        return 0.0, 0.0, 0.0

    ranked_results = deduplicate_by_source(results)

    def matching_label(result):
        for label in relevant_docs:
            if is_relevant(result, [label]):
                return label
        return None

    # Binary relevance of each unique source document.
    relevance = [
        1 if matching_label(result) is not None else 0
        for result in ranked_results
    ]

    # Recall@K: proportion of labelled relevant documents retrieved.
    found_labels = set()

    for result in ranked_results[:k]:
        label = matching_label(result)
        if label is not None:
            found_labels.add(label)

    recall = len(found_labels) / len(relevant_docs)

    # MRR: reciprocal rank of the first relevant result.
    mrr = 0.0

    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # DCG@K.
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance[:k], start=1)
    )

    # Ideal DCG@K for binary relevance.
    ideal_count = min(len(relevant_docs), k)

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # Do not silently cap nDCG. A value above 1 indicates a bug.
    if not 0.0 <= ndcg <= 1.0 + 1e-10:
        raise ValueError(
            f"Invalid nDCG value: {ndcg}. Check relevance labels "
            "and document deduplication."
        )

    return recall, mrr, ndcg


# --------------------------------------------------
# BM25-only retrieval
# --------------------------------------------------

def bm25_only(question, top_k=K):
    """Retrieve documents using BM25 keyword matching."""
    query_tokens = tokenize(question)
    scores = bm25_index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        record = indexed_records[index]
        metadata = record.get("metadata") or {}

        results.append({
            "id": record.get("id", str(index)),
            "document": record.get("document", ""),
            "metadata": metadata,
            "source": get_source(record),
            "score": float(scores[index]),
        })

    return results


# --------------------------------------------------
# Dense-only retrieval
# --------------------------------------------------

def dense_only(question, top_k=K):
    """Retrieve documents using dense vector similarity."""
    document_count = collection.count()

    if document_count == 0:
        return []

    query_embedding = embedding_model.encode(
        [BGE_QUERY_PREFIX + question],
        normalize_embeddings=True,
    ).tolist()

    response = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, document_count),
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results = []

    for index, record_id in enumerate(ids):
        metadata = metadatas[index] or {}
        document = documents[index] or ""

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        results.append({
            "id": record_id,
            "document": document,
            "metadata": metadata,
            "source": get_source({"metadata": metadata}),
            "score": (
                float(1.0 - distance)
                if distance is not None
                else 0.0
            ),
        })

    return results


# --------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------

def hybrid_retrieval(question, top_k=K):
    """
    Run the project's existing hybrid pipeline:
    BM25 + dense retrieval + RRF + reranking.
    """
    return retrieve(question, top_k=top_k)


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("No evaluation questions found.")

    methods = {
        "BM25-only": bm25_only,
        "Dense-only": dense_only,
        "Hybrid (RRF + reranking)": hybrid_retrieval,
    }

    totals = {
        name: [0.0, 0.0, 0.0]
        for name in methods
    }

    print("\nAGRICULTURE RAG RETRIEVAL COMPARISON")
    print("=" * 88)
    print(f"Evaluation questions: {len(test_cases)}")
    print(f"Metrics: Recall@{K}, MRR, nDCG@{K}")
    print("Evaluation unit: unique source documents")
    print("=" * 88)

    for question_number, test_case in enumerate(test_cases, start=1):
        question = test_case["question"]
        relevant_docs = test_case.get("relevant_docs", [])

        print(f"\n{question_number}. {question}")

        for method_name, method in methods.items():
            results = method(question, top_k=K)

            recall, mrr, ndcg = calculate_metrics(
                results,
                relevant_docs,
                k=K,
            )

            totals[method_name][0] += recall
            totals[method_name][1] += mrr
            totals[method_name][2] += ndcg

            print(
                f"   {method_name:<27}"
                f"Recall@{K}={recall:.3f}  "
                f"MRR={mrr:.3f}  "
                f"nDCG@{K}={ndcg:.3f}"
            )

    # Calculate macro averages over all evaluation questions.
    question_count = len(test_cases)

    print("\n" + "=" * 88)
    print("AVERAGE RESULTS")
    print("=" * 88)

    print(
        f"{'Method':<30}"
        f"{'Recall@5':>12}"
        f"{'MRR':>12}"
        f"{'nDCG@5':>12}"
    )
    print("-" * 66)

    for method_name, values in totals.items():
        averages = [
            value / question_count
            for value in values
        ]

        print(
            f"{method_name:<30}"
            f"{averages[0]:>12.3f}"
            f"{averages[1]:>12.3f}"
            f"{averages[2]:>12.3f}"
        )

    print("=" * 88)
    print("Comparison completed.")


if __name__ == "__main__":
    main()
