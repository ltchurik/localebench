import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localebench.api import ApiError, chat
from localebench.cli import main
from localebench.report import markdown_table
from localebench.runner import run_benchmark
from localebench.scorers import BuiltinScorer, LocaleLintScorer
from localebench.tasks import load_tasks


def test_perfect_model_scores_100():
    tasks = load_tasks(["sk", "fr"])
    by_prompt = {t["prompt"]: t["reference"] for t in tasks}
    res = run_benchmark(tasks, lambda p, s: by_prompt[p], [BuiltinScorer()], model="oracle")
    assert res["summary"]["overall"]["score"] == 1.0
    assert set(res["summary"]["by_lang"]) == {"sk", "fr"}


def test_bad_model_scores_0():
    tasks = load_tasks()
    by_prompt = {t["prompt"]: t["counterexample"] for t in tasks}
    res = run_benchmark(tasks, lambda p, s: by_prompt[p], [BuiltinScorer()], model="anglo")
    assert res["summary"]["overall"]["passed"] == 0


def test_api_errors_are_recorded_not_raised():
    def boom(p, s):
        raise ApiError("HTTP 500")

    res = run_benchmark(load_tasks(["de"]), boom, [BuiltinScorer()], model="down")
    assert res["summary"]["errors"] == 8 and res["summary"]["overall"]["passed"] == 0


def test_locale_lint_adapter_informational_and_strict():
    class FakeLint:
        @staticmethod
        def lint(text, locale):
            return [{"rule": "nbsp-percent"}] if "%" in text else []

    task = {"lang": "fr", "checks": []}
    info = LocaleLintScorer(module=FakeLint).score(task, "3,7 %")
    assert info[0]["ok"] is None and "nbsp-percent" in info[0]["detail"]
    strict = LocaleLintScorer(strict=True, module=FakeLint).score(task, "3,7 %")
    assert strict[0]["ok"] is False


def test_report_table():
    tasks = load_tasks(["sk"])
    ref = {t["prompt"]: t["reference"] for t in tasks}
    res = run_benchmark(tasks, lambda p, s: ref[p], [BuiltinScorer()], model="oracle")
    md = markdown_table([res])
    assert "`oracle`" in md and "| 100 |" in md


class _FakeOpenAI(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOpenAI.seen.append((self.path, self.headers.get("Authorization"), body))
        answer = "<think>hmm</think>Inflácia dosiahla 3,7 %."
        data = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_http_client_against_fake_endpoint(fake_server):
    out = chat("x", model="m", base_url=fake_server, api_key="k", system="sys")
    assert out == "Inflácia dosiahla 3,7 %."
    path, auth, body = _FakeOpenAI.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_cli_end_to_end(fake_server, tmp_path, capsys):
    rc = main(["run", "--model", "fake/m", "--base-url", fake_server, "--api-key", "k",
               "--langs", "sk", "--out", str(tmp_path), "--lint", "off", "-q"])
    assert rc == 0
    data = json.loads((tmp_path / "fake-m.json").read_text())
    ids = {t["id"]: t["passed"] for t in data["tasks"]}
    assert ids["sk-percent-005"] is True and ids["sk-date-001"] is False
    assert main(["report", "--results", str(tmp_path)]) == 0
    assert "`fake/m`" in capsys.readouterr().out
