"""
Uses an LLM to judge whether generated code actually satisfies the prompt
that produced it. Kept separate from the generation model config so you can
judge with a different (e.g. stronger) model than the one under test.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional

from ollama_client import chat

JUDGE_SYSTEM_PROMPT = """You are a strict code reviewer. You will be given:
  - the original user request
  - optional judging criteria describing what a correct solution must do
  - the code that was generated in response

Judge ONLY whether the code correctly and reasonably fulfills the request \
and criteria. Do not penalize style choices, missing docstrings, or minor \
naming differences. Do penalize: wrong/invented APIs, missing required \
behavior, logic errors, and ignoring explicit constraints in the request.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"correct": true or false, "score": integer 1-5, "reasoning": "one or two sentences"}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeVerdict:
    correct: Optional[bool]
    score: Optional[int]
    reasoning: str
    raw_response: str
    parse_error: str = ""


def judge_code(prompt: str, code: str, criteria: str = "", model: Optional[str] = None) -> JudgeVerdict:
    user_msg = (
        f"Original request:\n{prompt}\n\n"
        f"Judging criteria:\n{criteria or '(none provided; use general judgment)'}\n\n"
        f"Generated code:\n```\n{code}\n```\n\n"
        f"Return the JSON verdict now."
    )
    kwargs = {"temperature": 0.0}
    if model:
        kwargs["model"] = model
    raw = chat(user_msg, system=JUDGE_SYSTEM_PROMPT, **kwargs)

    match = _JSON_RE.search(raw)
    if not match:
        return JudgeVerdict(correct=None, score=None, reasoning="",
                             raw_response=raw, parse_error="No JSON object found in judge response")
    try:
        parsed = json.loads(match.group(0))
        return JudgeVerdict(
            correct=bool(parsed.get("correct")),
            score=int(parsed["score"]) if "score" in parsed else None,
            reasoning=str(parsed.get("reasoning", "")),
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return JudgeVerdict(correct=None, score=None, reasoning="",
                             raw_response=raw, parse_error=f"Failed to parse judge JSON: {e}")
