"""Build a leaderboard table from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "summary" in data and "model" in data:
            out.append(data)
    return out


def _pct(s: dict | None) -> str:
    if not s or s.get("score") is None:
        return "–"
    return f"{100 * s['score']:.0f}"


def markdown_table(results: list[dict]) -> str:
    if not results:
        return "_No results yet. Run `localebench run` first._\n"
    langs = sorted({lang for r in results for lang in r["summary"]["by_lang"]})
    cats = sorted({c for r in results for c in r["summary"]["by_category"]})
    results = sorted(results, key=lambda r: -(r["summary"]["overall"]["score"] or 0))
    head = ["Model", "Overall"] + langs
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        s = r["summary"]
        lines.append(
            "| " + " | ".join([f"`{r['model']}`", _pct(s["overall"])] + [_pct(s["by_lang"].get(x)) for x in langs]) + " |"
        )
    head2 = ["Model"] + cats
    lines += ["", "| " + " | ".join(head2) + " |", "|" + "---|" * len(head2)]
    for r in results:
        s = r["summary"]
        lines.append("| " + " | ".join([f"`{r['model']}`"] + [_pct(s["by_category"].get(c)) for c in cats]) + " |")
    lines.append("")
    lines.append("Scores are % of tasks passed. Each column is a different number of tasks; see results JSON.")
    return "\n".join(lines) + "\n"
