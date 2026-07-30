"""
Given a library name, find and download its documentation.

Strategy (first that yields content wins, but we try all and merge):
  1. llms.txt / llms-full.txt on the project's homepage domain
  2. PyPI project metadata -> project_urls (Docs / Homepage / Repository)
  3. GitHub repo README + /docs markdown files (via GitHub API, no auth needed
     for public repos at low request volumes)
  4. <name>.readthedocs.io as a fallback guess

Everything downloaded is cached to disk as {name}__{n}.json under
config.RAW_DOCS_DIR so re-runs don't re-fetch by default.
"""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": config.USER_AGENT}


@dataclass
class DocPage:
    library: str
    source_url: str
    title: str
    text: str          # cleaned markdown/plain text
    kind: str           # "llms_txt" | "readme" | "docs_md" | "html"


def _get(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp
    except requests.RequestException:
        pass
    return None


def _clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()
    # Prefer <main> or <article> if present -- less boilerplate
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _pypi_metadata(name: str) -> Optional[dict]:
    resp = _get(f"https://pypi.org/pypi/{name}/json")
    if not resp:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _github_repo_from_urls(urls: List[str]) -> Optional[str]:
    for u in urls:
        if "github.com" in u:
            parsed = urlparse(u)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return None


def _try_llms_txt(base_url: str, library: str) -> List[DocPage]:
    pages = []
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    for candidate in ("llms-full.txt", "llms.txt"):
        resp = _get(urljoin(root + "/", candidate))
        if resp and resp.text.strip():
            pages.append(DocPage(
                library=library,
                source_url=urljoin(root + "/", candidate),
                title=f"{library} llms.txt",
                text=resp.text,
                kind="llms_txt",
            ))
            break  # llms-full.txt takes priority over llms.txt
    return pages


def _fetch_github_docs(repo: str, library: str) -> List[DocPage]:
    """Pull README + markdown files under /docs from a public GitHub repo."""
    pages: List[DocPage] = []
    api_root = f"https://api.github.com/repos/{repo}"

    # README
    resp = _get(f"{api_root}/readme")
    if resp:
        try:
            meta = resp.json()
            dl = meta.get("download_url")
            if dl:
                raw = _get(dl)
                if raw:
                    pages.append(DocPage(
                        library=library, source_url=dl,
                        title=f"{repo} README", text=raw.text, kind="readme",
                    ))
        except ValueError:
            pass

    # /docs directory tree (shallow scan, markdown only)
    for docs_path in ("docs", "doc", "documentation"):
        resp = _get(f"{api_root}/contents/{docs_path}")
        if not resp:
            continue
        try:
            entries = resp.json()
        except ValueError:
            continue
        if not isinstance(entries, list):
            continue
        count = 0
        for entry in entries:
            if count >= config.MAX_PAGES_PER_LIBRARY:
                break
            if entry.get("type") == "file" and entry.get("name", "").endswith((".md", ".rst")):
                raw = _get(entry["download_url"])
                if raw:
                    pages.append(DocPage(
                        library=library, source_url=entry["download_url"],
                        title=entry["name"], text=raw.text, kind="docs_md",
                    ))
                    count += 1
        if pages:
            break  # found a docs folder, don't also try "doc" and "documentation"
    return pages


def _fetch_readthedocs(library: str) -> List[DocPage]:
    pages = []
    url = f"https://{library}.readthedocs.io/en/stable/"
    resp = _get(url)
    if not resp:
        url = f"https://{library}.readthedocs.io/en/latest/"
        resp = _get(url)
    if resp:
        text = _clean_html_to_text(resp.text)
        if len(text) > 200:
            pages.append(DocPage(
                library=library, source_url=url,
                title=f"{library} ReadTheDocs", text=text, kind="html",
            ))
    return pages


def resolve_and_fetch(library: str, force: bool = False) -> List[DocPage]:
    """
    Main entry point: given a library name, find and download its docs.
    Caches to disk; pass force=True to re-fetch.
    """
    cache_index = os.path.join(config.RAW_DOCS_DIR, f"{library}__index.json")
    if os.path.exists(cache_index) and not force:
        return _load_cached(library)

    pages: List[DocPage] = []

    meta = _pypi_metadata(library)
    project_urls = []
    homepage = None
    if meta:
        info = meta.get("info", {})
        homepage = info.get("home_page") or info.get("project_url")
        proj_urls = info.get("project_urls") or {}
        project_urls = list(proj_urls.values())
        if homepage:
            project_urls.append(homepage)

    # 1. llms.txt on any known project URL
    for u in project_urls:
        found = _try_llms_txt(u, library)
        if found:
            pages.extend(found)
            break

    # 2. GitHub README + docs
    repo = _github_repo_from_urls(project_urls)
    if repo:
        pages.extend(_fetch_github_docs(repo, library))

    # 3. ReadTheDocs fallback if we still have little content
    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 2000:
        pages.extend(_fetch_readthedocs(library))

    # 4. last resort: scrape the homepage itself
    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 500 and homepage:
        resp = _get(homepage)
        if resp:
            pages.append(DocPage(
                library=library, source_url=homepage,
                title=f"{library} homepage", text=_clean_html_to_text(resp.text),
                kind="html",
            ))

    _save_cache(library, pages)
    return pages


def _save_cache(library: str, pages: List[DocPage]) -> None:
    for i, p in enumerate(pages):
        path = os.path.join(config.RAW_DOCS_DIR, f"{library}__{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(p), f)
    index_path = os.path.join(config.RAW_DOCS_DIR, f"{library}__index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(pages), "fetched_at": time.time()}, f)


def _load_cached(library: str) -> List[DocPage]:
    index_path = os.path.join(config.RAW_DOCS_DIR, f"{library}__index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    pages = []
    for i in range(index["count"]):
        path = os.path.join(config.RAW_DOCS_DIR, f"{library}__{i}.json")
        with open(path, "r", encoding="utf-8") as f:
            pages.append(DocPage(**json.load(f)))
    return pages
