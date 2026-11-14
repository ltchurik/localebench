"""Built-in deterministic checks (the fallback scorer).

Each task carries a list of `checks`; a task passes only if all of them pass.

Types:
  regex        – re.search(value, answer) finds a match ("flags": "i" = ignore case)
  not_regex    – re.search finds no match
  contains     – answer contains the string (case-insensitive)
  not_contains – answer does not contain the string (case-insensitive)
  min_words    – answer has at least N words
  max_words    – answer has at most N words

Answers are normalized first: Unicode NFC, Markdown emphasis removed, and every
no-break / narrow no-break / thin space folded into a plain space. Spacing *type*
(e.g. U+202F before % in French) is the job of llm-locale-lint, not of these checks.
"""

from __future__ import annotations

import re
import unicodedata

CHECK_TYPES = {"regex", "not_regex", "contains", "not_contains", "min_words", "max_words"}

_SPACES = re.compile(r"[    ]")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<!\w)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = _SPACES.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _flags(check: dict) -> int:
    return re.IGNORECASE if "i" in check.get("flags", "") else 0


def run_check(check: dict, answer: str) -> tuple[bool, str]:
    """Return (passed, human-readable detail)."""
    t = check["type"]
    val = check.get("value", "")
    ans = normalize(answer)
    if t == "regex":
        return re.search(val, ans, _flags(check)) is not None, f"regex /{val}/"
    if t == "not_regex":
        return re.search(val, ans, _flags(check)) is None, f"not_regex /{val}/"
    if t == "contains":
        return normalize(val).casefold() in ans.casefold(), f"contains {val!r}"
    if t == "not_contains":
        return normalize(val).casefold() not in ans.casefold(), f"not_contains {val!r}"
    if t == "min_words":
        n = len(words(ans))
        return n >= int(val), f"min_words {val} (got {n})"
    if t == "max_words":
        n = len(words(ans))
        return n <= int(val), f"max_words {val} (got {n})"
    raise ValueError(f"unknown check type: {t}")
