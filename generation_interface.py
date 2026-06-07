from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from embedding_retrieval import RetrievedChunk, retrieve_top_k_chunks


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 5
DEFAULT_PERSIST_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "gwu_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CONFIDENCE_SIMILARITY = 0.32
FALLBACK_ANSWER = "I don't have enough information on that."


SYSTEM_PROMPT = """You answer questions using only the provided context.
Do not use outside knowledge, assumptions, or web knowledge.
If the context does not contain enough information to answer, reply with exactly:
I don't have enough information on that.

When you do answer, keep it concise and grounded in the retrieved documents.
Use bracket citations like [1], [2] that refer to the numbered context items.
Do not include section headers or a source list in your own response.
Do not mention any source that is not in the provided context.
"""


@dataclass(frozen=True)
class GroundedResponse:
    answer: str
    sources: list[RetrievedChunk]


def load_groq_client() -> object:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file before running this command."
        )

    from groq import Groq

    return Groq(api_key=api_key)


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {chunk.source_name}",
                    f"URL: {chunk.source_uri}",
                    f"Chunk: {chunk.chunk_index} | tokens {chunk.start_token}-{chunk.end_token} | similarity {chunk.similarity:.3f}",
                    f"Text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user_prompt = f"""Question: {question}

Context:
{context}

Respond in two sections exactly:
Answers:
- one or more concise grounded sentences or bullets with citations

Source list:
- [n] source_name | source_uri | chunk span | similarity
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def clean_model_answer(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines()]
    cleaned_lines: list[str] = []
    skipping_headers = {"answers:", "source list:", "## answers:", "## source list:"}

    for line in lines:
        if not line:
            continue
        lower_line = line.lower()
        if lower_line in skipping_headers:
            continue
        if lower_line.startswith("source list:"):
            break
        if re.match(r"^\s*-\s*\[\d+\]\s+", line):
            break
        cleaned_lines.append(line)

    cleaned_answer = "\n".join(cleaned_lines).strip()
    return cleaned_answer or FALLBACK_ANSWER


def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> GroundedResponse:
    chunks = retrieve_top_k_chunks(
        query=question,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=embedding_model,
    )

    if not chunks or max(chunk.similarity for chunk in chunks) < MIN_CONFIDENCE_SIMILARITY:
        return GroundedResponse(answer=FALLBACK_ANSWER, sources=[])

    client = load_groq_client()
    messages = build_messages(question, chunks)
    response = client.chat.completions.create(
        model=groq_model,
        messages=messages,
        temperature=0,
    )
    answer = clean_model_answer(response.choices[0].message.content or "")

    return GroundedResponse(answer=answer, sources=list(chunks))


def format_source_list(chunks: Sequence[RetrievedChunk]) -> str:
    lines: list[str] = ["Source list:"]
    if not chunks:
        lines.append("- No sources retrieved.")
        return "\n".join(lines)

    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"- [{index}] {chunk.source_name} | {chunk.source_uri} | chunk {chunk.chunk_index} | tokens {chunk.start_token}-{chunk.end_token} | similarity {chunk.similarity:.3f}"
        )
    return "\n".join(lines)


def format_response(response: GroundedResponse) -> str:
    return f"Answers:\n{response.answer}\n\n{format_source_list(response.sources)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded generation over retrieved GWU chunks.")
    parser.add_argument("question", help="Question to answer using only retrieved documents.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to retrieve before generation.")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR, help="ChromaDB persistence directory.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model used for retrieval.")
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL, help="Groq chat model to use for generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = generate_answer(
        question=args.question,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        groq_model=args.groq_model,
    )
    print(format_response(response))


if __name__ == "__main__":
    main()