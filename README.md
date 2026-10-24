# Agriculture RAG Assistant (adapted from the provided RAG retrieval project)

This version keeps the original architecture: Markdown ingestion and chunking, BGE embeddings, ChromaDB, BM25 sparse retrieval, dense retrieval, Reciprocal Rank Fusion (RRF), BGE cross-encoder reranking, LangChain, and Ollama. The sample knowledge base and prompts are agriculture-only.

## Requirements
- Python 3.10 or 3.11 recommended
- Ollama installed locally
- Internet access on first run to download embedding/reranker models and the Ollama model

## Windows PowerShell setup
```powershell
cd Agriculture_RAG_Adapted
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b
py -m src.ingest
streamlit run app_langchain.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that same terminal, then activate again. Keep Ollama running while using the app.

## Evaluation
After indexing, run `py -m src.evaluate`. Evaluation values are measured on this small sample test set; they are not claims of production performance.

## Notes
- First ingestion downloads `BAAI/bge-small-en-v1.5`; retrieval also downloads `BAAI/bge-reranker-base`.
- The app does not provide live weather, live mandi prices, or verified current government scheme status.
- The included agriculture documents are starter educational material. For real farmer decisions, verify crop- and region-specific guidance with official extension sources.
- The original query-intent code contains Kubernetes-specific special cases; agriculture questions fall through to the general retrieval path, so these cases do not drive agricultural ranking.
