"""localebench command line.

  localebench run --model gpt-4o-mini --base-url https://api.openai.com/v1 --langs sk,cs,de,pl,fr
  localebench report
  localebench validate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__
from .api import chat
from .report import load_results, markdown_table
from .runner import run_benchmark
from .scorers import build_scorers
from .tasks import available_langs, load_tasks, validate_tasks


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")


def _langs(arg: str | None, tasks_dir: Path | None) -> list[str]:
    if not arg or arg == "all":
        return available_langs(tasks_dir)
    return [x.strip() for x in arg.split(",") if x.strip()]


def cmd_run(a: argparse.Namespace) -> int:
    if not a.model or not a.base_url:
        print("error: --model and --base-url are required (or LOCALEBENCH_MODEL / LOCALEBENCH_BASE_URL)", file=sys.stderr)
        return 2
    tasks_dir = Path(a.tasks_dir) if a.tasks_dir else None
    tasks = load_tasks(_langs(a.langs, tasks_dir), tasks_dir)
    if a.limit:
        tasks = tasks[: a.limit]
    scorers = build_scorers(a.lint)

    def chat_fn(prompt: str, system: str | None) -> str:
        return chat(
            prompt,
            model=a.model,
            base_url=a.base_url,
            api_key=a.api_key,
            system=system,
            temperature=a.temperature,
            max_tokens=a.max_tokens,
            timeout=a.timeout,
        )

    def progress(rec: dict) -> None:
        mark = "ERR " if rec.get("error") else ("ok  " if rec["passed"] else "FAIL")
        print(f"{mark} {rec['id']}", file=sys.stderr)

    res = run_benchmark(
        tasks,
        chat_fn,
        scorers,
        model=a.model,
        workers=a.workers,
        meta={"temperature": a.temperature, "max_tokens": a.max_tokens},
        progress=None if a.quiet else progress,
    )
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug(a.model)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    s = res["summary"]
    print(f"\n{a.model}: {s['overall']['passed']}/{s['overall']['total']} tasks passed", end="")
    print(f" ({s['errors']} API errors)" if s["errors"] else "")
    for lang, v in s["by_lang"].items():
        print(f"  {lang}: {v['passed']}/{v['total']}")
    print(f"→ {path}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    results = load_results(Path(a.results))
    if a.format == "json":
        print(json.dumps([{"model": r["model"], **r["summary"]} for r in results], ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(markdown_table(results))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    tasks_dir = Path(a.tasks_dir) if a.tasks_dir else None
    tasks = load_tasks(None, tasks_dir)
    errors = validate_tasks(tasks)
    for e in errors:
        print(e, file=sys.stderr)
    print(f"{len(tasks)} tasks, {len(errors)} problems")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localebench", description="Do LLMs write like a native?")
    p.add_argument("--version", action="version", version=f"localebench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the benchmark against an OpenAI-compatible endpoint")
    r.add_argument("--model", default=os.environ.get("LOCALEBENCH_MODEL"))
    r.add_argument("--base-url", default=os.environ.get("LOCALEBENCH_BASE_URL"))
    r.add_argument("--api-key", default=None, help="default: $LOCALEBENCH_API_KEY or $OPENAI_API_KEY")
    r.add_argument("--langs", default="all", help="comma-separated, e.g. sk,cs,de,pl,fr (default: all)")
    r.add_argument("--tasks-dir", default=None)
    r.add_argument("--out", default="results")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--timeout", type=int, default=300)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    r.add_argument("--lint", choices=["auto", "off", "on", "strict"], default="auto",
                   help="llm-locale-lint scorer: auto=use if installed (informational)")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="leaderboard from results/*.json")
    rep.add_argument("--results", default="results")
    rep.add_argument("--format", choices=["md", "json"], default="md")
    rep.set_defaults(func=cmd_report)

    v = sub.add_parser("validate", help="check task files; references must pass, counterexamples must fail")
    v.add_argument("--tasks-dir", default=None)
    v.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
