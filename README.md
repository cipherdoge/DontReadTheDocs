# Library Docs RAG (Chroma + Ollama)

Give it a library name, it fetches and indexes the docs. Then ask it to
generate code, and it retrieves relevant doc chunks and grounds the LLM's
output in them.

## Setup

```bash
pip install -r requirements.txt

# Pull the models in Ollama
ollama pull qwen3.5:4b          # generation
ollama pull nomic-embed-text    # embeddings — see note below

ollama serve                     # if not already running
```

### Important note on models

`qwen3.5:4b` is a chat/generation model — it does **not** produce embeddings.
Ollama's `/api/embeddings` endpoint needs a model trained for that
(`nomic-embed-text`, `mxbai-embed-large`, `all-minilm`, etc.). This project
uses `qwen3.5:4b` for **code generation** and `nomic-embed-text` for
**embeddings**. Override either via env vars if you want different models:

```bash
export RAG_GEN_MODEL=qwen3.5:4b
export RAG_EMBED_MODEL=nomic-embed-text
```

## Usage

```bash
# 1. Index a library's docs
python cli.py ingest requests

# 2. Generate code against those docs
python cli.py ask requests "write a function that does a GET with retries and a timeout"

# Or go interactive for one library
python cli.py chat fastapi
```

Docs are cached under `data/raw_docs/` and embeddings under `data/chroma/`,
so re-running `ingest` for the same library is a no-op unless you pass
`--force`.

## How it works

1. **Resolution** — looks for an `llms.txt`/`llms-full.txt` on the project's
   site, then falls back to PyPI metadata → GitHub README + `/docs` folder,
   then ReadTheDocs, then the bare homepage.
2. **Chunking** — splits by markdown headers, and never splits a fenced code
   block across chunks.
3. **Storage** — Chroma, one collection with a `library` metadata filter per
   library (keeps API versions/libraries from bleeding into each other).
4. **Retrieval** — hybrid: Chroma vector search (via Ollama embeddings) +
   BM25 keyword search, merged with reciprocal rank fusion. BM25 matters
   because exact API/function names often beat semantic similarity for code
   docs.
5. **Generation** — retrieved chunks are stuffed into the prompt with source
   URLs, and the model is instructed to only use documented APIs and to cite
   which sections it used.

## Benchmarking

`benchmark.py` runs the full pipeline (ingest → retrieve → generate) against
a set of test cases defined in `benchmark_config.json`, and reports two
numbers:

- **Compile rate** — % of generated code that is syntactically valid.
  This is a *syntax* check only (`compile()` for Python, `node --check` for
  JS, `tsc --noEmit` for TS if installed) — it does not execute the code or
  install the target library, since running arbitrary LLM-generated code
  would need real sandboxing. If no checker exists for a case's language,
  that case is excluded from the compile-rate denominator.
- **Judged correctness** — an LLM judge (default: the same model as
  `judge_model` in the config) is shown the original prompt, optional
  `judge_criteria`, and the generated code, and returns a JSON verdict
  (`correct`, a 1–5 `score`, and reasoning).

```bash
python benchmark.py                                # uses benchmark_config.json
python benchmark.py --config my_cases.json
python benchmark.py --judge-model llama3.1:8b       # judge with a different/stronger model
python benchmark.py --skip-ingest                   # skip re-ingesting (already indexed)
```

Add test cases to `benchmark_config.json`:

```json
{
  "id": "requests_get_retries",
  "library": "requests",
  "language": "python",
  "prompt": "Write a function that does a GET with retries and a timeout",
  "judge_criteria": "Must use requests.get, must retry, must respect timeout"
}
```

Each run prints a per-case + summary report and saves the full detail
(including the raw generated response and raw judge output) as JSON under
`benchmark_reports/report_<timestamp>.json`.

**Judge caveat:** since the judge is itself an LLM, treat scores as a
useful signal for tracking regressions/improvements across pipeline
changes, not as ground truth. Spot-check a sample of `judge_reasoning`
against the actual code periodically.

## Known limitations

- No version pinning yet — re-ingesting overwrites the previous version's
  chunks for a library (upsert by chunk id, keyed on source URL, so stale
  chunks from removed pages aren't cleaned up automatically).
- GitHub doc-folder scan is shallow (top-level of `/docs` only, no
  recursion into subfolders).
- HTML scraping (ReadTheDocs/homepage fallback) is a generic boilerplate
  stripper — some sites will need site-specific selectors for cleaner text.
