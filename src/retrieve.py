
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "storage" / "chroma"

COLLECTION_NAME = "agriculture_knowledge"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

RETRIEVAL_TOP_K = 15
RERANK_TOP_K = 5
RRF_K = 60

BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Set to True only if you want to see more internal retrieval details.
DEBUG = False


# ---------------------------------------------------------
# Load the existing ChromaDB index
# ---------------------------------------------------------

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"ChromaDB storage was not found at: {CHROMA_PATH}\n"
        "Run your ingestion script first."
    )

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
    )

try:
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )
except Exception as exc:
    raise RuntimeError(
        f"Could not open ChromaDB collection '{COLLECTION_NAME}'. "
        "Check the collection name and run ingestion if necessary."
    ) from exc

if collection.count() == 0:
    raise RuntimeError(
        "The ChromaDB collection is empty. Run src\\ingest.py first."
    )


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

print("Loading BGE embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading BGE cross-encoder reranker...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)

print(f"Loaded collection with {collection.count()} chunks.")


# ---------------------------------------------------------
# Build a BM25 index from all indexed chunks
# ---------------------------------------------------------

def tokenize(text):
    """Simple, consistent tokenization for BM25."""
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


print("Loading indexed chunks for BM25...")

all_records = collection.get(
    include=["documents", "metadatas"]
)

all_documents = all_records.get("documents") or []
all_metadatas = all_records.get("metadatas") or []
all_ids = all_records.get("ids") or []

if not all_documents:
    raise RuntimeError(
        "No document text was found in the ChromaDB collection."
    )

bm25_corpus = [tokenize(doc or "") for doc in all_documents]
bm25_index = BM25Okapi(bm25_corpus)

# Retain the indexed records so BM25 results can be mapped to chunks.
indexed_records = []

for i, doc in enumerate(all_documents):
    metadata = (
        all_metadatas[i]
        if i < len(all_metadatas) and all_metadatas[i]
        else {}
    )

    indexed_records.append(
        {
            "id": all_ids[i] if i < len(all_ids) else str(i),
            "document": doc or "",
            "metadata": metadata,
        }
    )

print(f"BM25 index ready: {len(indexed_records)} chunks.")


# ---------------------------------------------------------
# Understand the query's troubleshooting intent
# ---------------------------------------------------------

def classify_query(query):
    """
    Identify the main troubleshooting topic.

    Explicit component names take precedence so a specific
    ingress/controller-manager question is not treated as generic.
    """
    q = query.lower()

    if (
        "ingress" in q
        or "ingress controller" in q
    ):
        return "ingress"

    if (
        "controller-manager" in q
        or "controller manager" in q
        or "kube-controller-manager" in q
    ):
        return "controller_manager"

    if (
        "admission webhook" in q
        or "webhook" in q
        or "x509" in q
        or "unknown authority" in q
        or "certificate signed by" in q
    ):
        return "webhook"

    if (
        "crashloopbackoff" in q
        or "crash loop" in q
        or "crashing" in q
    ):
        return "crashloop"

    if (
        "imagepullbackoff" in q
        or "image pull" in q
        or "errimagepull" in q
    ):
        return "imagepull"

    if (
        "api server" in q
        or "apiserver" in q
        or "connection refused" in q
    ):
        return "apiserver"

    return "general"


# ---------------------------------------------------------
# Source-path helpers and relevance adjustments
# ---------------------------------------------------------

def get_source(item):
    metadata = item.get("metadata") or {}

    return str(
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("filename")
        or metadata.get("path")
        or "unknown_source"
    ).replace("\\", "/")


def get_topic_adjustment(query, item, query_type):
    """
    Apply small, transparent topic adjustments after reranking.

    The cross-encoder remains the primary relevance scorer.
    """
    source = get_source(item).lower()
    text = item.get("document", "").lower()
    query_lower = query.lower()

    adjustment = 0.0

    if query_type == "crashloop":
        if "crashloopbackoff.md" in source:
            adjustment += 0.15

        if "init-container" in source or "init_container" in source:
            if "init container" in query_lower:
                adjustment += 0.20
            else:
                adjustment -= 0.10

    elif query_type == "ingress":
        if "ingress" in source or "ingress" in text:
            adjustment += 0.20

    elif query_type == "controller_manager":
        if (
            "controller-manager" in source
            or "controller_manager" in source
            or "kube-controller-manager" in text
        ):
            adjustment += 0.20

    elif query_type == "webhook":
        if "webhook" in source or "webhook" in text:
            adjustment += 0.20

        if (
            "x509" in query_lower
            or "unknown authority" in query_lower
        ):
            if "certificate" in source or "x509" in source:
                adjustment += 0.10

    elif query_type == "imagepull":
        if (
            "imagepullbackoff" in source
            or "image-pull" in source
            or "image pull" in text
        ):
            adjustment += 0.20

    elif query_type == "apiserver":
        if (
            "api-server" in source
            or "apiserver" in source
            or "api server" in text
        ):
            adjustment += 0.20

    return adjustment


# ---------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Combine ranked lists using:
        RRF(d) = sum(1 / (k + rank(d)))

    Ranks start at 1.
    """
    scores = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["id"]

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (k + rank)
            )

    return scores


# ---------------------------------------------------------
# Main retrieval pipeline
# ---------------------------------------------------------

def retrieve(query, top_k=RERANK_TOP_K):
    """
    Hybrid retrieval pipeline:
      1. BM25 sparse retrieval
      2. BGE dense retrieval
      3. Reciprocal Rank Fusion
      4. Query-intent filtering
      5. BGE cross-encoder reranking
      6. Source diversity to avoid duplicate chunks
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    query_type = classify_query(query)

    # -----------------------------
    # A. Sparse retrieval with BM25
    # -----------------------------

    query_tokens = tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:RETRIEVAL_TOP_K]

    bm25_results = []

    for idx in bm25_indices:
        record = indexed_records[idx]

        bm25_results.append(
            {
                "id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
                "bm25_score": float(bm25_scores[idx]),
            }
        )

    # -----------------------------
    # B. Dense retrieval with BGE
    # -----------------------------

    query_embedding = embedding_model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    dense_response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(RETRIEVAL_TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    dense_results = []

    ids_group = dense_response.get("ids") or [[]]
    docs_group = dense_response.get("documents") or [[]]
    metas_group = dense_response.get("metadatas") or [[]]
    distances_group = dense_response.get("distances") or [[]]

    dense_ids = ids_group[0] if ids_group else []
    dense_docs = docs_group[0] if docs_group else []
    dense_metas = metas_group[0] if metas_group else []
    dense_distances = distances_group[0] if distances_group else []

    for i, chunk_id in enumerate(dense_ids):
        dense_results.append(
            {
                "id": chunk_id,
                "document": (
                    dense_docs[i]
                    if i < len(dense_docs) and dense_docs[i]
                    else ""
                ),
                "metadata": (
                    dense_metas[i]
                    if i < len(dense_metas) and dense_metas[i]
                    else {}
                ),
                "dense_distance": (
                    float(dense_distances[i])
                    if i < len(dense_distances)
                    else None
                ),
            }
        )

    # -----------------------------
    # C. Fuse BM25 and dense results
    # -----------------------------

    rrf_scores = reciprocal_rank_fusion(
        [bm25_results, dense_results]
    )

    records_by_id = {}

    for item in bm25_results + dense_results:
        # Keep the most complete version of each chunk.
        if item["id"] not in records_by_id:
            records_by_id[item["id"]] = item
        else:
            existing = records_by_id[item["id"]]

            if not existing.get("document") and item.get("document"):
                existing["document"] = item["document"]

            if not existing.get("metadata") and item.get("metadata"):
                existing["metadata"] = item["metadata"]

    candidates = []

    for chunk_id, fusion_score in rrf_scores.items():
        item = records_by_id.get(chunk_id)

        if item is None:
            continue

        candidates.append(
            {
                "id": chunk_id,
                "document": item.get("document", ""),
                "metadata": item.get("metadata") or {},
                "rrf_score": float(fusion_score),
            }
        )

    candidates.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # -----------------------------
    # D. Query-intent filtering
    # -----------------------------

    # For a GENERIC CrashLoopBackOff question, avoid letting
    # component-specific ingress/controller-manager pages dominate.
    #
    # If the user explicitly asks about those components,
    # classify_query() returns a specific query type instead.
    if query_type == "crashloop":
        specialized_terms = (
            "ingress",
            "controller-manager",
            "controller_manager",
            "kube-controller-manager",
        )

        generic_candidates = [
            item
            for item in candidates
            if not any(
                term in get_source(item).lower()
                for term in specialized_terms
            )
        ]

        # Only apply the filter if relevant general candidates exist.
        if generic_candidates:
            candidates = generic_candidates

    # Avoid spending reranker time on a huge candidate set.
    candidates = candidates[: min(30, len(candidates))]

    if not candidates:
        return []

    # -----------------------------
    # E. Cross-encoder reranking
    # -----------------------------

    pairs = [
        [query, item["document"]]
        for item in candidates
    ]

    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    for item, rerank_score in zip(candidates, rerank_scores):
        item["rerank_score"] = float(rerank_score)

        item["topic_adjustment"] = get_topic_adjustment(
            query,
            item,
            query_type,
        )

        item["final_score"] = (
            item["rerank_score"] + item["topic_adjustment"]
        )

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    # -----------------------------
    # F. Select distinct source files
    # -----------------------------

    # Multiple chunks from the same file often repeat the same evidence.
    # Prefer one chunk per file in the final results.
    selected = []
    seen_sources = set()

    for item in candidates:
        source = get_source(item)

        if source in seen_sources:
            continue

        seen_sources.add(source)
        item["source"] = source
        selected.append(item)

        if len(selected) >= top_k:
            break

    # If there are too few distinct sources, allow additional chunks
    # from an already-selected source to fill the requested result count.
    if len(selected) < top_k:
        selected_ids = {item["id"] for item in selected}

        for item in candidates:
            if item["id"] in selected_ids:
                continue

            item["source"] = get_source(item)
            selected.append(item)
            selected_ids.add(item["id"])

            if len(selected) >= top_k:
                break

    if DEBUG:
        print(f"\nQuery type: {query_type}")
        print(f"BM25 candidates: {len(bm25_results)}")
        print(f"Dense candidates: {len(dense_results)}")
        print(f"Fused candidates: {len(candidates)}")

    return selected


# ---------------------------------------------------------
# Optional standalone test
# ---------------------------------------------------------

if __name__ == "__main__":
    question = input(
        "Enter an agriculture question: "
    ).strip()

    results = retrieve(question)

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    if not results:
        print("No relevant documents were retrieved.")

    for i, item in enumerate(results, start=1):
        print(f"\n{i}. {item['source']}")
        print(f"   Final score: {item['final_score']:.4f}")
        print(f"   Reranker score: {item['rerank_score']:.4f}")
        print(f"   Topic adjustment: {item['topic_adjustment']:.4f}")
        print(f"   Chunk ID: {item['id']}")
        print("-" * 60)
        print(item["document"][:900])
