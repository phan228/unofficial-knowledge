from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "gwu_chunks"
DEFAULT_PERSIST_DIR = Path("chroma_db")
DEFAULT_CHUNKS_FILE = Path("chunks.jsonl")
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class ChunkRecord:
    source_name: str
    source_uri: str
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source_name: str
    source_uri: str
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    text: str
    similarity: float


def load_chunks(chunks_file: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                ChunkRecord(
                    source_name=str(payload["source_name"]),
                    source_uri=str(payload["source_uri"]),
                    chunk_index=int(payload["chunk_index"]),
                    start_token=int(payload["start_token"]),
                    end_token=int(payload["end_token"]),
                    token_count=int(payload["token_count"]),
                    text=str(payload["text"]),
                )
            )
    return records


@lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "sentence-transformers is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return SentenceTransformer(model_name)


def get_client(persist_dir: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "chromadb is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    client = get_client(persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(
    chunks_file: Path = DEFAULT_CHUNKS_FILE,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    reset: bool = True,
) -> int:
    records = load_chunks(chunks_file)
    if not records:
        raise RuntimeError(f"No chunks found in {chunks_file}")

    client = get_client(persist_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = load_model(model_name)
    texts = [record.text for record in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    ids = [f"{record.source_name}:{record.chunk_index}:{record.start_token}-{record.end_token}" for record in records]
    metadatas = [
        {
            "source_name": record.source_name,
            "source_uri": record.source_uri,
            "chunk_index": record.chunk_index,
            "start_token": record.start_token,
            "end_token": record.end_token,
            "token_count": record.token_count,
        }
        for record in records
    ]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    return len(records)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[RetrievedChunk]:
    collection = get_collection(persist_dir, collection_name)
    model = load_model(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[RetrievedChunk] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - float(distance)
        retrieved.append(
            RetrievedChunk(
                source_name=str(metadata.get("source_name", "unknown")),
                source_uri=str(metadata.get("source_uri", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                start_token=int(metadata.get("start_token", 0)),
                end_token=int(metadata.get("end_token", 0)),
                token_count=int(metadata.get("token_count", 0)),
                text=str(document),
                similarity=similarity,
            )
        )

    return retrieved


def retrieve_top_k_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[RetrievedChunk]:
    """Return the top-k most relevant chunks for a query, including source metadata."""

    return retrieve(
        query=query,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )


def format_results(results: Sequence[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(results, start=1):
        lines.append(
            f"[{index}] {chunk.source_name} | similarity {chunk.similarity:.3f} | tokens {chunk.start_token}-{chunk.end_token}"
        )
        lines.append(f"    {chunk.source_uri}")
        lines.append(f"    {chunk.text}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a ChromaDB index over chunked GWU documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Embed chunks and store them in ChromaDB.")
    build_parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    build_parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    build_parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    build_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    build_parser.add_argument("--keep-existing", action="store_true", help="Do not delete the collection before writing.")

    query_parser = subparsers.add_parser("query", help="Run a similarity search over the stored chunks.")
    query_parser.add_argument("query", help="Search query to retrieve relevant chunks for.")
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query_parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    query_parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    query_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        count = build_index(
            chunks_file=args.chunks_file,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            reset=not args.keep_existing,
        )
        print(f"Indexed {count} chunks into {args.persist_dir} / {args.collection_name}")
        return

    if args.command == "query":
        results = retrieve(
            query=args.query,
            top_k=args.top_k,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
        )
        print(format_results(results))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()