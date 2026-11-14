"""Task loading and validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .checks import CHECK_TYPES, run_check

REQUIRED = ("id", "lang", "category", "prompt", "checks")

SYSTEM_PROMPTS = {
    "sk": "Odpovedaj po slovensky.",
    "cs": "Odpovídej česky.",
    "de": "Antworte auf Deutsch.",
    "pl": "Odpowiadaj po polsku.",
    "fr": "Réponds en français.",
}


def default_tasks_dir() -> Path:
    """Bundled tasks in an installed wheel, else ./tasks in a source checkout."""
    bundled = Path(__file__).parent / "tasks"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[2] / "tasks"


def available_langs(tasks_dir: Path | None = None) -> list[str]:
    d = tasks_dir or default_tasks_dir()
    return sorted(p.stem for p in d.glob("*.jsonl"))


def load_tasks(langs: list[str] | None = None, tasks_dir: Path | None = None) -> list[dict]:
    d = tasks_dir or default_tasks_dir()
    langs = langs or available_langs(d)
    tasks: list[dict] = []
    for lang in langs:
        path = d / f"{lang}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no tasks for language {lang!r} ({path})")
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{n}: {e}") from e
    return tasks


def validate_tasks(tasks: list[dict]) -> list[str]:
    """Return a list of problems. Also asserts that each task's `reference` answer passes
    and its `counterexample` fails, so every check is proven to discriminate."""
    errors: list[str] = []
    seen: set[str] = set()
    for t in tasks:
        tid = t.get("id", "<no id>")
        for k in REQUIRED:
            if k not in t:
                errors.append(f"{tid}: missing {k!r}")
        if tid in seen:
            errors.append(f"{tid}: duplicate id")
        seen.add(tid)
        for c in t.get("checks", []):
            if c.get("type") not in CHECK_TYPES:
                errors.append(f"{tid}: unknown check type {c.get('type')!r}")
            elif c["type"] in ("regex", "not_regex"):
                try:
                    re.compile(c["value"])
                except re.error as e:
                    errors.append(f"{tid}: bad regex {c['value']!r}: {e}")
        if errors:
            continue
        ref = t.get("reference")
        if ref is not None:
            for c in t["checks"]:
                ok, detail = run_check(c, ref)
                if not ok:
                    errors.append(f"{tid}: reference answer fails {detail}")
        bad = t.get("counterexample")
        if bad is not None and all(run_check(c, bad)[0] for c in t["checks"]):
            errors.append(f"{tid}: counterexample passes all checks")
    return errors
