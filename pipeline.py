"""
High-level operations:
  ingest_library(name)          -> fetch, chunk, embed, store docs for a library
  generate_code(library, ask)   -> retrieve relevant chunks, ask the LLM to
                                    write code grounded in them
"""

from __future__ import annotations
from typing import List

import doc_fetcher
import chunker
from vector_store import LibraryStore
from ollama_client import chat

SYSTEM_PROMPT = """You are a precise coding assistant. You are given documentation \
excerpts for a specific library, retrieved by a search system. Use ONLY the APIs, \
function names, and patterns shown in the provided context. Do not invent methods, \
parameters, or classes that do not appear in the context.

If the context is insufficient to fully answer the request, say so explicitly and \
generate the best partial solution you can from what's given, clearly marking any \
assumptions.

After the code, list which context sections you drew from (by source URL)."""


def ingest_library(library: str, force: bool = False) -> dict:
    """Fetch docs for `library`, chunk them, embed, and store in Chroma."""
    pages = doc_fetcher.resolve_and_fetch(library, force=force)
    if not pages:
        return {"library": library, "pages": 0, "chunks": 0, "status": "no_docs_found"}

    chunks = chunker.chunk_pages(pages)
    store = LibraryStore()
    store.add_chunks(chunks)

    return {
        "library": library,
        "pages": len(pages),
        "chunks": len(chunks),
        "sources": sorted({p.source_url for p in pages}),
        "status": "ok",
    }


def _format_context(results: List[dict]) -> str:
    blocks = []
    for r in results:
        meta = r["metadata"]
        header = f"[Source: {meta.get('source_url')} | Section: {meta.get('section_title')}]"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_code(library: str, request: str, auto_ingest: bool = True) -> str:
    """
    Retrieve relevant doc chunks for `library` and ask the LLM to generate
    code for `request`, grounded in those chunks.
    """
    store = LibraryStore()
    if not store.has_library(library):
        if not auto_ingest:
            return (f"No docs stored for '{library}' yet. Run ingest_library('{library}') first."
                     )
        result = ingest_library(library)
        if result["status"] != "ok":
            return (f"Could not find or fetch documentation for '{library}'. "
                     f"Try a different name or check the spelling.")

    results = store.hybrid_search(request, library=library)
    if not results:
        return (f"No relevant documentation found for '{library}' matching your request. "
                 f"Try rephrasing, or the ingested docs may not cover this.")

    context = _format_context(results)
    prompt = (
        f"Library: {library}\n\n"
        f"Documentation context:\n{context}\n\n"
        f"---\n\nUser request: {request}\n\n"
        f"Write the code now."
    )
    return chat(prompt, system=SYSTEM_PROMPT)
