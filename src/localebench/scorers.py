"""Scorer interface.

A scorer looks at one (task, answer) pair and returns a list of check results:
  {"scorer": str, "type": str, "ok": bool | None, "detail": str}

`ok=None` means informational: it is reported but does not decide pass/fail.

Built in:
  BuiltinScorer   – runs the task's own deterministic `checks` (always on).
  LocaleLintScorer – optional; uses llm-locale-lint (https://github.com/ltchurik/llm-locale-lint)
                     to flag generic locale issues in any answer. Informational by default,
                     pass/fail with --lint strict.
"""

from __future__ import annotations

from typing import Any, Protocol

from .checks import run_check


class Scorer(Protocol):
    name: str

    def score(self, task: dict, answer: str) -> list[dict]: ...


class BuiltinScorer:
    name = "builtin"

    def score(self, task: dict, answer: str) -> list[dict]:
        out = []
        for c in task["checks"]:
            ok, detail = run_check(c, answer)
            out.append({"scorer": self.name, "type": c["type"], "ok": ok, "detail": detail})
        return out


def _field(issue: Any, *names: str) -> Any:
    for n in names:
        if isinstance(issue, dict) and n in issue:
            return issue[n]
        if hasattr(issue, n):
            return getattr(issue, n)
    return None


class LocaleLintScorer:
    """Adapter for llm-locale-lint. Uses `llm_locale_lint.lint(text, locale=...)`, which returns Issue objects with `rule_id`."""

    name = "llm-locale-lint"

    def __init__(self, strict: bool = False, module: Any = None):
        if module is None:
            import llm_locale_lint as module  # noqa: PLC0415 – optional dependency
        if not hasattr(module, "lint"):
            raise ImportError("llm_locale_lint has no lint() function")
        self._lint = module.lint
        self.strict = strict

    def score(self, task: dict, answer: str) -> list[dict]:
        issues = list(self._lint(answer, locale=task["lang"]))
        rules = [str(_field(i, "rule_id", "rule", "code", "id") or "issue") for i in issues]
        detail = f"{len(issues)} issue(s)" + (f": {', '.join(sorted(set(rules)))}" if rules else "")
        ok: bool | None = (not issues) if self.strict else None
        return [{"scorer": self.name, "type": "lint", "ok": ok, "detail": detail, "issues": len(issues)}]


def build_scorers(lint: str = "auto") -> list[Scorer]:
    """lint: 'off' | 'auto' (use if installed, informational) | 'on' (required) | 'strict' (required, affects pass)."""
    scorers: list[Scorer] = [BuiltinScorer()]
    if lint == "off":
        return scorers
    try:
        scorers.append(LocaleLintScorer(strict=(lint == "strict")))
    except ImportError:
        if lint in ("on", "strict"):
            raise SystemExit(
                "llm-locale-lint is not installed: pip install git+https://github.com/ltchurik/llm-locale-lint"
            ) from None
    return scorers
