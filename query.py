from __future__ import annotations

from pathlib import Path

from generation_interface import FALLBACK_ANSWER, generate_answer


DEFAULT_PERSIST_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "gwu_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 5


def ask(question: str) -> dict[str, object]:
    response = generate_answer(
        question=question,
        top_k=DEFAULT_TOP_K,
        persist_dir=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        groq_model=DEFAULT_GROQ_MODEL,
    )

    source_lines = [
        f"[{index}] {chunk.source_name} | {chunk.source_uri} | chunk {chunk.chunk_index} | tokens {chunk.start_token}-{chunk.end_token} | similarity {chunk.similarity:.3f}"
        for index, chunk in enumerate(response.sources, start=1)
    ]

    return {
        "answer": response.answer or FALLBACK_ANSWER,
        "sources": source_lines,
    }