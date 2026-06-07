from __future__ import annotations

import argparse
import html
import json
import importlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

import requests


DEFAULT_CHUNK_SIZE = 250
DEFAULT_OVERLAP = 50

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
HTML_EXTENSIONS = {".html", ".htm"}
PDF_EXTENSION = ".pdf"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
REQUEST_TIMEOUT_SECONDS = 30
HTML_CONTENT_SELECTORS = ("article", "main", "[role='main']", "body")
UNWANTED_HTML_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "option",
    "textarea",
    "aside",
    "nav",
    "footer",
)
SOURCE_DOMAIN_ALIASES = {
    "ratemyprofessors.com": "ratemyprofessors.com",
    "www.ratemyprofessors.com": "ratemyprofessors.com",
    "reddit.com": "reddit.com",
    "www.reddit.com": "reddit.com",
    "old.reddit.com": "reddit.com",
    "talk.collegeconfidential.com": "talk.collegeconfidential.com",
    "unigo.com": "unigo.com",
    "www.unigo.com": "unigo.com",
}

TOKEN_PATTERN = re.compile(r"\S+")
DOCUMENTS_SECTION_PATTERN = re.compile(
    r"## Documents\s*(.*?)\n## Chunking Strategy", re.DOTALL | re.IGNORECASE
)
DOCUMENT_ROW_PATTERN = re.compile(
    r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*(https?://[^|\s]+)\s*\|$",
    re.MULTILINE,
)
FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((?:[^)]+)\)")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?", re.MULTILINE)
MARKDOWN_LIST_PATTERN = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")
BOILERPLATE_LINE_PATTERNS = (
    re.compile(r"(?i)^\s*(cookie(s)?( settings)?|accept all cookies|manage cookies|privacy policy|terms of use)\s*$"),
    re.compile(r"(?i)read more|show more|load more|continue reading"),
    re.compile(r"(?i)share( this)?|copy link|comment(s)?|comment count|reply|react|like|follow us|subscribe"),
    re.compile(r"(?i)advertisement|sponsored|promoted|ad choices|all rights reserved|back to top|skip to content"),
    re.compile(r"(?i)^\s*(george washington university|gw reviews|the unofficial guide|unofficial guide|reviews)\s*$"),
    re.compile(r"(?i)^\s*(copy the code|write a review|enter to win|get started|quick facts|student life|college reviews|college reviews faqs|student body|tuition & aid|about george washington university|student life reviews|george washington university reviews|what's your overall opinion of george washington university\??|what's your overall opinion of this school\??|what would you rate on-campus housing\??|how would you rate on-campus housing\??|how would you rate off-campus housing\??|how would you rate campus food\??|how would you rate campus facilities\??|how would you rate class sizes\??|how would you rate school activities\??|how would you rate local services\??|how would you rate academics\??|what should every freshman at your school know before they start\??|describe the students at your school\.?|what is your overall opinion of this school\??|what are the academics like at your school\??|what are the most popular student activities/groups\??|what is the stereotype of students at your school\??|is the stereotype of students at your school accurate\??|here's your chance: say anything about your college\??|describe how your school looks to someone who's never seen it\??|what do you consider the worst thing about your school\??|what's unique about your campus\??|what makes you want to attend\??|what sets this school apart from others\??|what's the most frustrating thing about your school\??|what kind of person should not attend this school\??|what do you brag about most when you tell your friends about your school\??|what's the one thing you wish someone had told you about freshman year\??|what kind of person should attend this school\??|describe your favorite campus traditions\??)\s*$"),
    re.compile(r"(?i)^\s*(don't see the professor you're looking for\?|add a professor|read all \d+ answers|what is your overall opinion of this school\?|what do you think about this school\?)\s*$"),
    re.compile(r"(?i)^\s*\d+\s+students? rated\s+.*$"),
    re.compile(r"(?i)^\s*\d{4,}\s*$"),
    re.compile(r"(?i)^\s*skip to main content.*$"),
    re.compile(r"(?i)^\s*log in\s*/?create account.*$"),
    re.compile(r"(?i)^\s*(menu|search|sign in|log in|register|my account|home|about|contact)\s*$"),
    re.compile(r"(?i)^[\|·•/\-\s]*(home|about|academics|admissions|research|student life|news|events|faculty|staff|students)[\|·•/\-\s]*$"),
)


@dataclass(frozen=True)
class SourceDocument:
    source_name: str
    source_uri: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source_name: str
    source_uri: str
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    text: str


@dataclass(frozen=True)
class SourceSpec:
    source_name: str
    url: str


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS | HTML_EXTENSIONS | {PDF_EXTENSION}:
            yield path


def extract_planning_sources(planning_file: Path) -> list[SourceSpec]:
    planning_text = planning_file.read_text(encoding="utf-8", errors="ignore")
    documents_section_match = DOCUMENTS_SECTION_PATTERN.search(planning_text)
    if not documents_section_match:
        return []

    section_text = documents_section_match.group(1)
    sources: list[SourceSpec] = []
    for row_match in DOCUMENT_ROW_PATTERN.finditer(section_text):
        source_name = row_match.group(1).strip()
        url = row_match.group(3).strip()
        sources.append(SourceSpec(source_name=source_name, url=url))
    return sources


def decode_response_body(response: requests.Response) -> str:
    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    return response.text


def html_to_visible_text(raw_html: str) -> str:
    try:
        bs4 = importlib.import_module("bs4")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "HTML parsing requires beautifulsoup4. Install it with `pip install beautifulsoup4`."
        ) from exc

    BeautifulSoup = bs4.BeautifulSoup
    Comment = bs4.Comment

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag_name in UNWANTED_HTML_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()

    container = None
    for selector in HTML_CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            break
    if container is None:
        container = soup

    text = container.get_text("\n", strip=True)
    return text


def normalize_hostname(url: str) -> str:
    hostname = urlparse(url).netloc.lower()
    return SOURCE_DOMAIN_ALIASES.get(hostname, hostname)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slice_text_between_markers(text: str, start_markers: Sequence[str], end_markers: Sequence[str]) -> str:
    lower_text = text.lower()
    start_index = 0
    for marker in start_markers:
        marker_index = lower_text.find(marker.lower())
        if marker_index != -1:
            start_index = marker_index
            break

    end_index = len(text)
    for marker in end_markers:
        marker_index = lower_text.find(marker.lower(), start_index + 1)
        if marker_index != -1:
            end_index = min(end_index, marker_index)

    return text[start_index:end_index]


def extract_ratemyprofessors_text(raw_html: str) -> str:
    try:
        bs4 = importlib.import_module("bs4")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "HTML parsing requires beautifulsoup4. Install it with `pip install beautifulsoup4`."
        ) from exc

    BeautifulSoup = bs4.BeautifulSoup
    Comment = bs4.Comment

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag_name in UNWANTED_HTML_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()

    cards: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href*='/professor/']"):
        text = collapse_whitespace(anchor.get_text(" ", strip=True))
        if not text:
            continue
        normalized = text.lower()
        if "advertisement" in normalized or "add a professor" in normalized:
            continue
        if "help" in normalized or "guidelines" in normalized or "privacy policy" in normalized:
            continue
        if len(text.split()) < 4:
            continue

        parent = anchor.find_parent(["article", "li", "div"])
        if parent is not None:
            parent_text = collapse_whitespace(parent.get_text(" ", strip=True))
            if len(parent_text) > len(text):
                text = parent_text

        normalized = text.lower()
        if any(token in normalized for token in ("advertisement", "add a professor", "site guidelines")):
            continue
        if text in seen:
            continue
        seen.add(text)
        cards.append(text)

    if cards:
        return "\n\n".join(cards)

    return html_to_visible_text(raw_html)


def extract_unigo_text(raw_html: str) -> str:
    visible_text = html_to_visible_text(raw_html)
    visible_text = slice_text_between_markers(
        visible_text,
        start_markers=("GEORGE WASHINGTON UNIVERSITY REVIEWS", "STUDENT LIFE REVIEWS"),
        end_markers=(
            "GEORGE WASHINGTON UNIVERSITY FAQS",
            "STUDENT BODY",
            "TUITION & AID",
            "Additional Links",
            "TOP STUDENT RATED COLLEGES",
        ),
    )
    visible_text = re.sub(r"(?<=[\w.,;:!?])nn(?=[A-Z])", "\n\n", visible_text)
    return visible_text


def extract_html_text(raw_html: str, url: str) -> str:
    hostname = normalize_hostname(url)
    if hostname == "ratemyprofessors.com":
        return extract_ratemyprofessors_text(raw_html)
    if hostname == "unigo.com":
        return extract_unigo_text(raw_html)
    return html_to_visible_text(raw_html)


def fetch_url(url: str) -> tuple[str, str, str]:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        final_url = response.url
        return decode_response_body(response), content_type, final_url
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def fetch_json(url: str) -> tuple[object, str]:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json,text/plain,*/*",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json(), response.url
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def build_reddit_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") + "/.json"
    query = parsed.query
    if query:
        query = f"{query}&raw_json=1&limit=100"
    else:
        query = "raw_json=1&limit=100"
    return parsed._replace(path=path, query=query).geturl()


def build_college_confidential_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") + ".json"
    query = parsed.query
    return parsed._replace(path=path, query=query).geturl()


def extract_reddit_text(url: str) -> tuple[str, str]:
    json_url = build_reddit_json_url(url)
    payload, final_url = fetch_json(json_url)
    texts: list[str] = []

    if isinstance(payload, dict):
        children = payload.get("data", {}).get("children", [])
        for child in children:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            title = collapse_whitespace(str(data.get("title", "")))
            selftext = collapse_whitespace(str(data.get("selftext", "")))
            score = data.get("score")
            comments = data.get("num_comments")
            subreddit = data.get("subreddit_name_prefixed") or data.get("subreddit")

            parts = [part for part in (title, selftext) if part]
            if not parts:
                continue

            if subreddit:
                parts.insert(0, str(subreddit))
            if score is not None or comments is not None:
                parts.append(f"score {score} comments {comments}")

            entry = "\n".join(parts)
            if not looks_like_low_quality_chunk(entry):
                texts.append(entry)

    if texts:
        return "\n\n".join(texts), final_url

    return html_to_visible_text(str(payload)), final_url


def extract_college_confidential_text(url: str) -> tuple[str, str]:
    json_url = build_college_confidential_json_url(url)
    payload, final_url = fetch_json(json_url)
    texts: list[str] = []

    if isinstance(payload, dict):
        topic_list = payload.get("topic_list", {})
        topics = topic_list.get("topics", [])
        for topic in topics:
            if not isinstance(topic, dict):
                continue

            title = collapse_whitespace(str(topic.get("fancy_title") or topic.get("title") or ""))
            excerpt = collapse_whitespace(str(topic.get("excerpt") or ""))
            tags = topic.get("tags") or []
            replies = topic.get("posts_count")
            views = topic.get("views")
            last_activity = topic.get("last_posted_at") or topic.get("created_at")

            parts = [part for part in (title, excerpt) if part]
            if tags:
                parts.append("tags: " + ", ".join(str(tag) for tag in tags))
            if replies is not None or views is not None:
                parts.append(f"replies {replies} views {views}")
            if last_activity:
                parts.append(str(last_activity))

            entry = "\n".join(parts)
            if entry and not looks_like_low_quality_chunk(entry):
                texts.append(entry)

    if texts:
        return "\n\n".join(texts), final_url

    return html_to_visible_text(str(payload)), final_url


def extract_text_from_url(url: str) -> tuple[str, str]:
    hostname = normalize_hostname(url)

    if hostname == "reddit.com":
        return extract_reddit_text(url)

    if hostname == "talk.collegeconfidential.com":
        return extract_college_confidential_text(url)

    raw_body, content_type, final_url = fetch_url(url)
    suffix = Path(final_url.split("?", 1)[0]).suffix.lower()

    if content_type in {"text/html", "application/xhtml+xml"} or suffix in HTML_EXTENSIONS:
        return extract_html_text(raw_body, final_url), final_url

    if content_type.startswith("text/") or suffix in TEXT_EXTENSIONS:
        return raw_body, final_url

    if suffix == PDF_EXTENSION or content_type == "application/pdf":
        raise RuntimeError(
            f"Remote PDF sources are not supported by this script yet: {url}."
        )

    return raw_body, final_url


def read_pdf(path: Path) -> str:
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading PDF files requires pdfplumber. Install it with `pip install pdfplumber`."
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html_file(path: Path) -> str:
    return html_to_visible_text(path.read_text(encoding="utf-8", errors="ignore"))


def remove_consecutive_duplicate_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    previous_line = None
    for line in lines:
        if line and line != previous_line:
            deduped.append(line)
        previous_line = line
    return deduped


def looks_like_boilerplate(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()
    if not normalized:
        return True
    if any(pattern.search(normalized) for pattern in BOILERPLATE_LINE_PATTERNS):
        return True
    if normalized.startswith(("http://", "https://", "www.")):
        return True
    if normalized.count("|") >= 2:
        return True
    nav_tokens = {
        "home",
        "about",
        "admissions",
        "academics",
        "research",
        "students",
        "student life",
        "news",
        "events",
        "faculty",
        "staff",
        "contact",
        "menu",
        "search",
        "sign in",
        "log in",
        "register",
    }
    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z\-']*", normalized)]
    if len(words) >= 2 and all(word in nav_tokens for word in words):
        return True
    if len(normalized) <= 2:
        return True
    return False


def looks_like_low_quality_chunk(text: str) -> bool:
    normalized = collapse_whitespace(text)
    if not normalized:
        return True

    lower = normalized.lower()
    if any(marker in lower for marker in ("advertisement", "add a professor", "privacy policy", "terms of use", "additional links")):
        return True

    tokens = tokenize(normalized)
    if len(tokens) < 20:
        return True

    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    if not alpha_tokens:
        return True
    if len(alpha_tokens) / len(tokens) < 0.6:
        return True

    unique_ratio = len({token.lower() for token in alpha_tokens}) / len(alpha_tokens)
    if unique_ratio < 0.35:
        return True

    return False


def extract_raw_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return read_text_file(path)
    if suffix in HTML_EXTENSIONS:
        return read_html_file(path)
    if suffix == PDF_EXTENSION:
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_documents_from_planning(planning_file: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    failures: list[str] = []
    for source in extract_planning_sources(planning_file):
        try:
            raw_text, final_url = extract_text_from_url(source.url)
            cleaned = clean_text(raw_text, Path(final_url).suffix.lower())
            if cleaned:
                documents.append(
                    SourceDocument(source_name=source.source_name, source_uri=final_url, text=cleaned)
                )
        except Exception as exc:
            failures.append(f"{source.source_name} ({source.url}): {exc}")

    for failure in failures:
        print(f"Warning: {failure}")

    return documents


def clean_text(raw_text: str, suffix: str) -> str:
    text = raw_text.replace("\x00", "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = FRONT_MATTER_PATTERN.sub("", text)

    if suffix in TEXT_EXTENSIONS:
        text = CODE_FENCE_PATTERN.sub("\n", text)
        text = MARKDOWN_IMAGE_PATTERN.sub(r"\1", text)
        text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
        text = MARKDOWN_HEADING_PATTERN.sub("", text)
        text = MARKDOWN_BLOCKQUOTE_PATTERN.sub("", text)
        text = MARKDOWN_LIST_PATTERN.sub("", text)

    text = WHITESPACE_PATTERN.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_BLANK_PATTERN.sub("\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not looks_like_boilerplate(line)]
    lines = remove_consecutive_duplicate_lines(lines)
    text = "\n".join(lines)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenize(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append((start, end, " ".join(chunk_tokens)))
        if end == len(tokens):
            break
        start += step

    return chunks


def load_local_documents(root: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in iter_source_files(root):
        raw_text = extract_raw_text(path)
        cleaned = clean_text(raw_text, path.suffix.lower())
        if cleaned:
            documents.append(SourceDocument(source_name=path.name, source_uri=str(path), text=cleaned))
    return documents


def build_chunks(documents: Iterable[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunk_index = 0
        for start_token, end_token, text in chunk_text(document.text, chunk_size=chunk_size, overlap=overlap):
            if looks_like_low_quality_chunk(text):
                continue
            chunks.append(
                Chunk(
                    source_name=document.source_name,
                    source_uri=document.source_uri,
                    chunk_index=chunk_index,
                    start_token=start_token,
                    end_token=end_token,
                    token_count=end_token - start_token,
                    text=text,
                )
            )
            chunk_index += 1
    return chunks


def write_chunks_jsonl(chunks: Iterable[Chunk], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "source_name": chunk.source_name,
                        "source_uri": chunk.source_uri,
                        "chunk_index": chunk.chunk_index,
                        "start_token": chunk.start_token,
                        "end_token": chunk.end_token,
                        "token_count": chunk.token_count,
                        "text": chunk.text,
                    },
                    ensure_ascii=True,
                )
            )
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch planning.md links and/or local documents, clean them, and split them into overlapping token chunks."
    )
    parser.add_argument(
        "--planning-file",
        type=Path,
        default=Path("planning.md"),
        help="Planning file whose Documents section contains the source URLs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional local documents directory to ingest alongside planning-file URLs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Where to write the chunk metadata and text as JSONL.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Target chunk size in whitespace tokens.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--preview-document",
        action="store_true",
        help="Print the first cleaned document so you can inspect the cleaning step.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        default=0,
        help="Print this many representative chunks after chunking.",
    )
    return parser.parse_args()


def preview_text(label: str, text: str, max_chars: int = 2500) -> None:
    print(f"\n--- {label} ---")
    if len(text) > max_chars:
        print(text[:max_chars])
        print(f"... [truncated {len(text) - max_chars} chars]")
    else:
        print(text)


def select_representative_chunks(chunks: Sequence[Chunk], count: int) -> list[Chunk]:
    if count <= 0 or not chunks:
        return []
    if count >= len(chunks):
        return list(chunks)
    if count == 1:
        return [chunks[len(chunks) // 2]]

    positions = []
    last_index = len(chunks) - 1
    for offset in range(count):
        position = round(offset * last_index / (count - 1))
        positions.append(position)

    unique_positions = []
    seen = set()
    for position in positions:
        if position not in seen:
            seen.add(position)
            unique_positions.append(position)

    if len(unique_positions) < count:
        for position in range(len(chunks)):
            if position not in seen:
                seen.add(position)
                unique_positions.append(position)
            if len(unique_positions) == count:
                break

    return [chunks[position] for position in sorted(unique_positions[:count])]


def main() -> None:
    args = parse_args()

    if not args.planning_file.exists():
        raise FileNotFoundError(f"Planning file does not exist: {args.planning_file}")

    documents = load_documents_from_planning(args.planning_file)
    if args.input_dir is not None:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
        documents.extend(load_local_documents(args.input_dir))

    if not documents:
        raise RuntimeError("No documents were loaded from the planning links or local directory.")

    if args.preview_document:
        first_document = documents[0]
        preview_text(
            f"Cleaned document from {first_document.source_name} ({first_document.source_uri})",
            first_document.text,
        )

    chunks = build_chunks(documents, chunk_size=args.chunk_size, overlap=args.overlap)

    if args.preview_chunks > 0:
        preview_chunks = select_representative_chunks(chunks, args.preview_chunks)
        for chunk in preview_chunks:
            preview_text(
                f"Chunk {chunk.chunk_index} from {chunk.source_name} ({chunk.source_uri}) | tokens {chunk.start_token}-{chunk.end_token}",
                chunk.text,
            )

    write_chunks_jsonl(chunks, args.output_file)

    print(f"Loaded {len(documents)} documents from planning links and local files")
    print(f"Wrote {len(chunks)} chunks to {args.output_file}")

if __name__ == "__main__":
    main()