from localebench.checks import normalize, run_check
from localebench.tasks import available_langs, load_tasks, validate_tasks


def test_seed_languages_present():
    assert {"sk", "cs", "de", "pl", "fr"} <= set(available_langs())


def test_tasks_valid_and_discriminating():
    tasks = load_tasks()
    assert len(tasks) >= 40
    assert validate_tasks(tasks) == []


def test_every_task_has_reference_and_counterexample():
    for t in load_tasks():
        assert t.get("reference") and t.get("counterexample"), t["id"]


def test_normalize_folds_nbsp_variants():
    assert normalize("3,7 % a 1 234,50 €") == "3,7 % a 1 234,50 €"


def test_check_types():
    assert run_check({"type": "regex", "value": "a+b"}, "xaab")[0]
    assert not run_check({"type": "not_regex", "value": "x"}, "x")[0]
    assert run_check({"type": "contains", "value": "AHOJ"}, "ahoj svet")[0]
    assert run_check({"type": "min_words", "value": 2}, "dve slová")[0]
    assert not run_check({"type": "max_words", "value": 1}, "dve slová")[0]
