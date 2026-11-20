"""Unit tests for curation, split integrity, and leakage detection."""
import json
from pathlib import Path

import pytest

from src.curation import (
    CONFUSERS,
    SEED_TEMPLATES,
    build_corpus,
    fill_slots,
    normalize,
)
from src.config import (
    CONFUSER_KIND_SET,
    ESCALATE_LABEL,
    INTENT_LABELS,
    LABEL_SET,
)
from src.dataset import load_jsonl, validate_split
from src.leakage import assert_no_leakage, exact_overlap, report as leakage_report


def test_normalize_is_canonical():
    assert normalize("  Hello   World!  ") == "hello world"
    assert normalize("Summarize, PLEASE.") == "summarize please"
    assert normalize("don't") == normalize("DON'T")


def test_fill_slots_is_seeded():
    import random

    t = "translate this into {lang}"
    a = fill_slots(t, random.Random(7))
    b = fill_slots(t, random.Random(7))
    assert a == b
    assert a.split()[-1] in {"spanish", "french", "german", "japanese", "arabic", "urdu", "portuguese"}


def test_every_intent_has_templates():
    for label in INTENT_LABELS:
        assert label in SEED_TEMPLATES and len(SEED_TEMPLATES[label]) >= 5


def test_every_confuser_kind_has_pool():
    for kind in CONFUSER_KIND_SET:
        assert kind in CONFUSERS and len(CONFUSERS[kind]) >= 5


def test_corpus_is_reproducible():
    c1 = build_corpus(seed=20261119)
    c2 = build_corpus(seed=20261119)
    assert [r["text"] for r in c1.train] == [r["text"] for r in c2.train]
    assert [r["text"] for r in c1.test] == [r["text"] for r in c2.test]
    assert c1.stats["train_sha256"] == c2.stats["train_sha256"]


def test_corpus_split_is_disjoint():
    corpus = build_corpus(seed=20261119)
    train_keys = {normalize(r["text"]) for r in corpus.train}
    test_keys = {normalize(r["text"]) for r in corpus.test}
    assert not (train_keys & test_keys)


def test_corpus_covers_all_labels_on_both_sides():
    corpus = build_corpus(seed=20261119)
    for name, rows in (("train", corpus.train), ("test", corpus.test)):
        labels = {r["label"] for r in rows}
        assert labels == LABEL_SET, f"{name} missing {LABEL_SET - labels}"


def test_confuser_kinds_present_on_both_sides():
    corpus = build_corpus(seed=20261119)
    for name, rows in (("train", corpus.train), ("test", corpus.test)):
        kinds = {r["kind"] for r in rows}
        assert CONFUSER_KIND_SET <= kinds


def test_confusers_all_labelled_escalate():
    corpus = build_corpus(seed=20261119)
    for r in corpus.train + corpus.test:
        if r["kind"] != "intent":
            assert r["label"] == ESCALATE_LABEL


def test_families_do_not_span_splits():
    """The leakage guarantee at the family level: no template family on both sides."""
    corpus = build_corpus(seed=20261119)
    train_fams = {r["family"] for r in corpus.train}
    test_fams = {r["family"] for r in corpus.test}
    assert not (train_fams & test_fams)


def test_no_near_duplicate_leakage():
    corpus = build_corpus(seed=20261119)
    rep = leakage_report(corpus.train, corpus.test)
    assert rep["passed"], json.dumps(rep["near_duplicate_examples"], indent=2)


def test_written_files_roundtrip(tmp_path: Path):
    from src.curation import write_corpus

    stats = write_corpus(tmp_path, seed=20261119)
    train = load_jsonl(tmp_path / "train.jsonl")
    test = load_jsonl(tmp_path / "test.jsonl")
    assert len(train) == stats["train_rows"]
    assert len(test) == stats["test_rows"]
    validate_split(train, test)
    assert_no_leakage(train, test)
    on_disk = json.loads((tmp_path / "curation_report.json").read_text())
    assert on_disk == stats


def test_validate_split_rejects_thin_test_sets():
    from src.dataset import validate_split

    rows = [{"text": f"t{i}", "label": "code", "kind": "intent", "family": ""} for i in range(3)]
    with pytest.raises(ValueError):
        validate_split(rows, rows[:1])


def test_load_jsonl_rejects_bad_labels(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "x", "label": "not_a_label"}) + "\n")
    with pytest.raises(ValueError):
        load_jsonl(p)
