"""
Splits DocPage text into retrieval-sized chunks.

Rules:
  - Never split inside a fenced code block (```...```).
  - Prefer splitting on markdown headers (#, ##, ###) so a chunk is a
    coherent section rather than an arbitrary character window.
  - If a section is still too big, fall back to char-based splitting with
    overlap, but only on prose -- code blocks are kept whole even if that
    means a chunk exceeds CHUNK_MAX_CHARS.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

import config
from doc_fetcher import DocPage

HEADER_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Chunk:
    id: str
    library: str
    source_url: str
    section_title: str
    text: str
    has_code: bool


def _split_into_sections(text: str) -> List[tuple]:
    """Split on markdown headers; returns list of (title, body)."""
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append((title, body))
    return sections


def _split_prose_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return parts


def _chunk_section(title: str, body: str, max_chars: int, overlap: int) -> List[str]:
    """
    Keep code fences intact. Split the surrounding prose around them, then
    reassemble so no chunk cuts a ``` block in half.
    """
    code_blocks = CODE_FENCE_RE.findall(body)
    if not code_blocks:
        return _split_prose_with_overlap(body, max_chars, overlap)

    # Interleave prose and code as ordered segments
    segments = []
    last_end = 0
    for m in CODE_FENCE_RE.finditer(body):
        prose = body[last_end:m.start()]
        if prose.strip():
            segments.append(("prose", prose))
        segments.append(("code", m.group(0)))
        last_end = m.end()
    trailing = body[last_end:]
    if trailing.strip():
        segments.append(("prose", trailing))

    # Greedily pack segments into chunks up to max_chars; never split a code segment
    chunks, current = [], ""
    for kind, seg in segments:
        if kind == "code":
            if len(current) + len(seg) > max_chars and current:
                chunks.append(current)
                current = ""
            current += ("\n" if current else "") + seg
        else:
            for piece in _split_prose_with_overlap(seg, max_chars, overlap):
                if len(current) + len(piece) > max_chars and current:
                    chunks.append(current)
                    current = ""
                current += ("\n" if current else "") + piece
    if current:
        chunks.append(current)
    return chunks


def chunk_page(page: DocPage) -> List[Chunk]:
    sections = _split_into_sections(page.text)
    chunks: List[Chunk] = []
    idx = 0
    for title, body in sections:
        if not body.strip():
            continue
        for piece in _chunk_section(title, body, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS):
            if not piece.strip():
                continue
            cid = f"{page.library}::{page.source_url}::{idx}"
            chunks.append(Chunk(
                id=cid,
                library=page.library,
                source_url=page.source_url,
                section_title=title or page.title,
                text=piece.strip(),
                has_code="```" in piece,
            ))
            idx += 1
    return chunks


def chunk_pages(pages: List[DocPage]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in pages:
        all_chunks.extend(chunk_page(p))
    return all_chunks
