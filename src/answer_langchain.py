
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

try:
    from .retrieve import retrieve
except ImportError:
    from retrieve import retrieve


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an agriculture information assistant. Answer the user's question using only the supplied agricultural documents as your primary evidence.

Rules:
1. Give clear, practical, plain-language guidance.
2. Use retrieved evidence and cite the supporting source filenames.
3. Do not invent live weather, mandi prices, scheme status, local conditions, or field observations.
4. Agricultural recommendations depend on crop, variety, region, season, soil, and growth stage. Ask follow-up questions when needed.
5. Do not claim a confirmed pest or disease diagnosis from limited symptoms.
6. Do not prescribe pesticide dose, chemical mixtures, or treatment schedules unless the retrieved evidence clearly supports the specific crop and issue; direct the user to the approved product label and local extension advice.
7. If retrieved evidence is insufficient, say so and recommend a suitable local agriculture extension officer, Krishi Vigyan Kendra, soil testing lab, or plant clinic.
8. Distinguish general information from a crop-specific recommendation.

Retrieved agricultural documentation:
{context}
""",
        ),
        (
            "human",
            "Agriculture question:\n{question}\n\nProvide a useful answer grounded in the supplied documents.",
        ),
    ]
)


def format_context(results, max_docs=4, max_chars=1800):
    """Format retrieved documents as evidence for the LLM."""

    if not results:
        return "No relevant documentation was retrieved."

    sections = []
    seen_paths = set()

    for item in results:
        path = (
            item.get("path")
            or item.get("source")
            or "Unknown source"
        )

        if path in seen_paths:
            continue

        document = (
            item.get("document")
            or item.get("text")
            or item.get("content")
            or ""
        ).strip()

        if not document:
            continue

        seen_paths.add(path)

        sections.append(
            f"Source: {path}\n"
            f"Documentation:\n{document[:max_chars]}"
        )

        if len(sections) >= max_docs:
            break

    if not sections:
        return "Retrieved results contained no readable document text."

    return "\n\n---\n\n".join(sections)


def generate_answer(question):
    """Generate an evidence-grounded answer using LangChain and Ollama."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter a troubleshooting question.")

    # Reuse the existing hybrid retrieval pipeline:
    # BM25 + BGE dense retrieval + RRF + BGE reranking.
    results = retrieve(question)

    # Prepare the retrieved documentation for the LLM.
    context = format_context(results)

    # Connect LangChain to the local Ollama model.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    # LangChain chain:
    # Prompt -> ChatOllama -> String output parser.
    chain = PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return answer, results
