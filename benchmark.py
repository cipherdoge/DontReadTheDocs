"""
Benchmarks the full RAG pipeline (ingest -> retrieve -> generate) against a
set of test cases defined in a config file.

For each test case:
  1. Ensures the library's docs are ingested.
  2. Calls pipeline.generate_code(library, prompt).
  3. Extracts the main code block from the response.
  4. Checks whether it's syntactically valid ("compiles") for its language.
  5. Asks an LLM judge whether the code actually satisfies the prompt.

Reports:
  - Compile rate: % of test cases producing syntactically valid code
    (only counted over languages we have a checker for)
  - Correctness rate: % of test cases the judge marked correct
  - A per-case breakdown, plus a saved JSON + markdown report.

Usage:
    python benchmark.py                              # uses benchmark_config.json
    python benchmark.py --config my_cases.json
    python benchmark.py --judge-model llama3.1:8b     # judge with a different model
    python benchmark.py --skip-ingest                 # assume libraries already ingested
"""

from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import pipeline
import code_utils
import llm_judge

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "benchmark_config.json")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "benchmark_reports")


@dataclass
class CaseResult:
    id: str
    library: str
    language: str
    prompt: str
    generated_response: str
    extracted_code: str
    compile_checked: bool
    compiles: Optional[bool]
    compile_error: str
    judge_correct: Optional[bool]
    judge_score: Optional[int]
    judge_reasoning: str
    judge_parse_error: str
    duration_seconds: float
    error: str = ""


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_case(case: dict, judge_model: Optional[str], skip_ingest: bool) -> CaseResult:
    library = case["library"]
    language = case.get("language", "python")
    cid = case.get("id", f"{library}-{hash(case['prompt']) % 10000}")
    prompt = case["prompt"]
    criteria = case.get("judge_criteria", "")

    t0 = time.time()
    try:
        if not skip_ingest:
            pipeline.ingest_library(library)
        response = pipeline.generate_code(library, prompt)
    except Exception as e:
        return CaseResult(
            id=cid, library=library, language=language, prompt=prompt,
            generated_response="", extracted_code="", compile_checked=False,
            compiles=None, compile_error="", judge_correct=None, judge_score=None,
            judge_reasoning="", judge_parse_error="",
            duration_seconds=time.time() - t0, error=f"Pipeline error: {e}",
        )

    block = code_utils.pick_main_code_block(response, language)
    code = block.code if block else ""

    if not code:
        compile_result = code_utils.CompileResult(checked=False, compiles=None, error="No code block found")
    else:
        compile_result = code_utils.check_compiles(code, language)

    if code:
        verdict = llm_judge.judge_code(prompt, code, criteria=criteria, model=judge_model)
    else:
        verdict = llm_judge.JudgeVerdict(correct=False, score=1, reasoning="No code was generated.",
                                          raw_response="", parse_error="")

    return CaseResult(
        id=cid, library=library, language=language, prompt=prompt,
        generated_response=response, extracted_code=code,
        compile_checked=compile_result.checked, compiles=compile_result.compiles,
        compile_error=compile_result.error,
        judge_correct=verdict.correct, judge_score=verdict.score,
        judge_reasoning=verdict.reasoning, judge_parse_error=verdict.parse_error,
        duration_seconds=time.time() - t0,
    )


def summarize(results: List[CaseResult]) -> dict:
    total = len(results)
    compile_checked = [r for r in results if r.compile_checked]
    compiled_ok = [r for r in compile_checked if r.compiles]
    judged = [r for r in results if r.judge_correct is not None]
    judged_correct = [r for r in judged if r.judge_correct]
    scores = [r.judge_score for r in results if r.judge_score is not None]

    return {
        "total_cases": total,
        "compile_checked_cases": len(compile_checked),
        "compile_rate": (len(compiled_ok) / len(compile_checked)) if compile_checked else None,
        "judged_cases": len(judged),
        "correctness_rate": (len(judged_correct) / len(judged)) if judged else None,
        "average_judge_score": (sum(scores) / len(scores)) if scores else None,
        "errors": sum(1 for r in results if r.error),
    }


def print_report(results: List[CaseResult], summary: dict) -> None:
    print("\n" + "=" * 70)
    print("PER-CASE RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\n[{r.id}]  library={r.library}  language={r.language}  ({r.duration_seconds:.1f}s)")
        if r.error:
            print(f"  ERROR: {r.error}")
            continue
        compile_str = "n/a (no checker)" if not r.compile_checked else ("PASS" if r.compiles else f"FAIL ({r.compile_error[:80]})")
        print(f"  Compiles:  {compile_str}")
        if r.judge_parse_error:
            print(f"  Judge:     could not parse verdict ({r.judge_parse_error})")
        else:
            print(f"  Judge:     correct={r.judge_correct}  score={r.judge_score}/5")
            print(f"             {r.judge_reasoning}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total test cases:     {summary['total_cases']}")
    if summary["compile_rate"] is not None:
        print(f"Compile rate:         {summary['compile_rate']*100:.1f}%  "
              f"({summary['compile_checked_cases']} cases had a syntax checker)")
    else:
        print("Compile rate:         n/a (no cases had a syntax checker)")
    if summary["correctness_rate"] is not None:
        print(f"Judged correctness:   {summary['correctness_rate']*100:.1f}%  "
              f"({summary['judged_cases']} cases judged)")
    else:
        print("Judged correctness:   n/a")
    if summary["average_judge_score"] is not None:
        print(f"Average judge score:  {summary['average_judge_score']:.2f}/5")
    if summary["errors"]:
        print(f"Pipeline errors:      {summary['errors']}")
    print("=" * 70 + "\n")


def save_report(results: List[CaseResult], summary: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"report_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Benchmark the library-docs RAG pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to benchmark config JSON")
    parser.add_argument("--judge-model", default=None, help="Ollama model to use as judge (overrides config)")
    parser.add_argument("--skip-ingest", action="store_true",
                         help="Skip re-ingesting libraries (assume already indexed)")
    args = parser.parse_args()

    config = load_config(args.config)
    test_cases = config["test_cases"]
    judge_model = args.judge_model or config.get("judge_model")

    print(f"Running {len(test_cases)} test case(s)...")
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {case.get('id', case['library'])}...")
        results.append(run_case(case, judge_model=judge_model, skip_ingest=args.skip_ingest))

    summary = summarize(results)
    print_report(results, summary)
    path = save_report(results, summary)
    print(f"Full report saved to: {path}")


if __name__ == "__main__":
    main()
