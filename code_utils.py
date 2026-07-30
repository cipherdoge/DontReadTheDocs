"""
Utilities shared by the benchmark:
  - extract_code_blocks: pull fenced code blocks out of an LLM response
  - check_compiles: language-specific syntax validity check

NOTE: "compiles" here means "is syntactically valid", not "runs correctly
against the real library". We deliberately do NOT execute generated code
(that would require sandboxing and installing arbitrary dependencies) --
this is a syntax-level compile check only. Semantic correctness is left to
the LLM judge.
"""

from __future__ import annotations
import re
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional

FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@dataclass
class CodeBlock:
    language: str  # as declared on the fence, lowercased; "" if not declared
    code: str


def extract_code_blocks(text: str) -> List[CodeBlock]:
    blocks = []
    for m in FENCE_RE.finditer(text):
        lang = m.group(1).strip().lower()
        code = m.group(2)
        if code.strip():
            blocks.append(CodeBlock(language=lang, code=code))
    return blocks


def pick_main_code_block(text: str, expected_language: str) -> Optional[CodeBlock]:
    """Prefer a fenced block tagged with the expected language; fall back to
    the largest fenced block; fall back to None if there's no code at all."""
    blocks = extract_code_blocks(text)
    if not blocks:
        return None

    lang_aliases = {
        "python": {"python", "py", "python3"},
        "javascript": {"javascript", "js", "jsx", "node"},
        "typescript": {"typescript", "ts", "tsx"},
    }
    aliases = lang_aliases.get(expected_language.lower(), {expected_language.lower()})

    matching = [b for b in blocks if b.language in aliases]
    if matching:
        return max(matching, key=lambda b: len(b.code))
    return max(blocks, key=lambda b: len(b.code))


@dataclass
class CompileResult:
    checked: bool          # whether a checker exists for this language
    compiles: Optional[bool]  # None if not checked
    error: str = ""


def check_compiles(code: str, language: str) -> CompileResult:
    language = language.lower()
    if language in ("python", "py", "python3"):
        return _check_python(code)
    if language in ("javascript", "js", "jsx", "node"):
        return _check_node(code, ext=".js")
    if language in ("typescript", "ts", "tsx"):
        return _check_node(code, ext=".ts", typescript=True)
    return CompileResult(checked=False, compiles=None, error=f"No syntax checker for language '{language}'")


def _check_python(code: str) -> CompileResult:
    try:
        compile(code, "<generated>", "exec")
        return CompileResult(checked=True, compiles=True)
    except SyntaxError as e:
        return CompileResult(checked=True, compiles=False, error=str(e))


def _check_node(code: str, ext: str, typescript: bool = False) -> CompileResult:
    if typescript:
        # `node --check` doesn't understand TS syntax; needs tsc if available.
        if subprocess_which("tsc") is None:
            return CompileResult(checked=False, compiles=None,
                                  error="tsc not installed; cannot syntax-check TypeScript")
        checker_cmd = ["tsc", "--noEmit"]
    else:
        checker_cmd = ["node", "--check"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(checker_cmd + [path], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return CompileResult(checked=True, compiles=True)
        return CompileResult(checked=True, compiles=False, error=result.stderr.strip())
    except FileNotFoundError:
        return CompileResult(checked=False, compiles=None, error=f"'{checker_cmd[0]}' not installed")
    except subprocess.TimeoutExpired:
        return CompileResult(checked=True, compiles=False, error="syntax check timed out")
    finally:
        os.unlink(path)


def subprocess_which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)
