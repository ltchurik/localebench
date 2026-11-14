"""Run tasks against a model and aggregate scores."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from . import __version__
from .api import ApiError
from .scorers import Scorer
from .tasks import SYSTEM_PROMPTS

# chat_fn(prompt, system) -> answer text. Injected so tests and custom backends need no HTTP.
ChatFn = Callable[[str, "str | None"], str]


def eval_task(task: dict, chat_fn: ChatFn, scorers: list[Scorer]) -> dict:
    rec = {"id": task["id"], "lang": task["lang"], "category": task["category"], "prompt": task["prompt"]}
    system = task.get("system", SYSTEM_PROMPTS.get(task["lang"]))
    t0 = time.time()
    try:
        answer = chat_fn(task["prompt"], system)
    except ApiError as e:
        rec.update(answer=None, passed=False, error=str(e), checks=[], latency=round(time.time() - t0, 2))
        return rec
    rec["latency"] = round(time.time() - t0, 2)
    rec["answer"] = answer
    checks = [r for s in scorers for r in s.score(task, answer)]
    rec["checks"] = checks
    rec["passed"] = all(c["ok"] is not False for c in checks)
    return rec


def summarize(records: list[dict]) -> dict:
    def rate(rs: list[dict]) -> dict:
        n = len(rs)
        p = sum(1 for r in rs if r["passed"])
        return {"passed": p, "total": n, "score": round(p / n, 4) if n else None}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["lang"]].append(r)
        by_cat[r["category"]].append(r)
    return {
        "overall": rate(records),
        "errors": sum(1 for r in records if r.get("error")),
        "by_lang": {k: rate(v) for k, v in sorted(by_lang.items())},
        "by_category": {k: rate(v) for k, v in sorted(by_cat.items())},
    }


def run_benchmark(
    tasks: list[dict],
    chat_fn: ChatFn,
    scorers: list[Scorer],
    *,
    model: str,
    workers: int = 4,
    meta: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for rec in ex.map(lambda t: eval_task(t, chat_fn, scorers), tasks):
            records.append(rec)
            if progress:
                progress(rec)
    return {
        "model": model,
        "localebench_version": __version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorers": [s.name for s in scorers],
        **(meta or {}),
        "summary": summarize(records),
        "tasks": records,
    }
