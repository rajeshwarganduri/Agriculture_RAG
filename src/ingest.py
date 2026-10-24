
from pathlib import Path
import re

import yaml
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = ROOT / "data" / "documents"
CHROMA_DIR = ROOT / "storage" / "chroma"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 32

COLLECTION_NAME = "agriculture_knowledge"


# ---------------------------------------------------------
# Clean text
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize common text encoding issues and whitespace."""

    replacements = {
        "â€“": "–",
        "â€”": "—",
        "Â·": "·",
        "\ufeff": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text.strip()


# ---------------------------------------------------------
# Parse Markdown and YAML metadata
# ---------------------------------------------------------

def parse_markdown(path: Path):
    """Read Markdown and extract YAML front matter when present."""

    raw = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )

    raw = clean_text(raw)
    metadata = {}

    match = re.match(
        r"\A---\s*\n(.*?)\n---\s*\n?",
        raw,
        re.S,
    )

    if match:
        try:
            parsed = yaml.safe_load(match.group(1))

            if isinstance(parsed, dict):
                metadata = parsed

        except yaml.YAMLError:
            print(f"Warning: Could not parse metadata in {path.name}")

        raw = raw[match.end():]

    return metadata, raw.strip()


# ---------------------------------------------------------
# Split text into overlapping chunks
# ---------------------------------------------------------

def chunk_text(text: str):
    """Create overlapping character-based chunks."""

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer paragraph or line boundaries when possible.
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
            )

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move backwards to preserve overlap between chunks.
        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


# ---------------------------------------------------------
# Prepare ChromaDB-compatible metadata
# ---------------------------------------------------------

def safe_metadata(metadata, path, chunk_index):
    """Create metadata using supported scalar values."""

    title = metadata.get(
        "title",
        path.stem.replace("-", " ").title(),
    )

    category = metadata.get(
        "category",
        path.parent.name,
    )

    return {
        "title": str(title),
        "category": str(category),
        "source": str(path.relative_to(DOCUMENTS_DIR)),
        "chunk_index": int(chunk_index),
    }


# ---------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------

def main():

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    files = sorted(DOCUMENTS_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError(
            "No Markdown documents were found."
        )

    print(f"Found {len(files)} Markdown documents.")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    # Open persistent ChromaDB storage.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Rebuild this collection to avoid duplicate chunks on reruns.
    # WARNING: This deletes the existing collection first.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Agriculture knowledge base",
            "hnsw:space": "cosine",
        },
    )

    pending_ids = []
    pending_texts = []
    pending_metadata = []

    # Read documents and create chunks.
    for file_number, path in enumerate(files, start=1):

        metadata, text = parse_markdown(path)
        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):

            chunk_id = (
                f"{path.relative_to(DOCUMENTS_DIR).as_posix()}"
                f"::{chunk_index}"
            )

            pending_ids.append(chunk_id)
            pending_texts.append(chunk)

            pending_metadata.append(
                safe_metadata(
                    metadata,
                    path,
                    chunk_index,
                )
            )

        if file_number % 50 == 0:
            print(
                f"Read {file_number}/{len(files)} documents..."
            )

    total_chunks = len(pending_texts)

    if total_chunks == 0:
        raise RuntimeError(
            "No text chunks were created from the documents."
        )

    print(f"Created {total_chunks} chunks.")
    print("Generating embeddings and storing them in ChromaDB...")

    # Embed and store chunks in manageable batches.
    indexed_count = 0

    for start in range(0, total_chunks, BATCH_SIZE):

        end = start + BATCH_SIZE

        texts = pending_texts[start:end]

        embeddings = model.encode(
            [
                "Represent this sentence for searching relevant passages: "
                + text
                for text in texts
            ],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=pending_ids[start:end],
            documents=texts,
            metadatas=pending_metadata[start:end],
            embeddings=embeddings,
        )

        indexed_count += len(texts)

        print(
            f"Indexed {indexed_count}/{total_chunks} chunks."
        )

    # Final status.
    print("\nIngestion completed successfully.")
    print(f"Documents read: {len(files)}")
    print(f"Chunks indexed: {collection.count()}")
    print(f"Persistent database: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
