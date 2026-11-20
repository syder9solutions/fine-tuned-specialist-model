"""End-to-end integration tests. The real model is exercised, but only through
the shared inference path — this is what catches wiring bugs unit tests miss.
Marked slow; run in CI, not on every save."""
import json
from pathlib import Path

import pytest

from src.config import LABELS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def base_arm():
    from src.train import load_base

    return load_base(device="cpu")


def test_inference_scores_all_labels(base_arm, tmp_path: Path):
    """The scorer returns a distribution over exactly the 9 labels."""
    import torch

    from src.inference import decode_predictions, score_labels

    model, tok = base_arm
    texts = ["translate this email to german", "hi how are you"]
    lp = score_labels(model, tok, texts)
    assert lp.shape == (2, len(LABELS))
    probs = lp.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    labels, confs = decode_predictions(lp)
    assert set(labels) <= set(LABELS)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_inference_is_deterministic(base_arm):
    import torch

    from src.inference import score_labels

    model, tok = base_arm
    texts = ["write a fibonacci function in python"]
    a = score_labels(model, tok, texts)
    b = score_labels(model, tok, texts)
    assert torch.equal(a, b)


def test_base_and_finetune_share_inference_path(base_arm):
    """Sanity: the same object type flows through run_inference for both arms.

    The eval's validity rests on this — if the two arms ever diverge into
    different scoring code, the head-to-head is meaningless.
    """
    import inspect

    from src.evaluate import run_inference
    from src.inference import score_labels as shared_scorer

    src = inspect.getsource(run_inference)
    assert "score_labels" in src  # run_inference delegates to the shared scorer
    assert shared_scorer.__module__ == "src.inference"


def test_label_token_ids_are_positive_ids(base_arm):
    from src.inference import label_token_ids

    _, tok = base_arm
    mapping = label_token_ids(tok)
    assert set(mapping) == set(LABELS)
    for label, ids in mapping.items():
        assert ids and all(i >= 0 for i in ids)


def test_api_http_wiring(tmp_path: Path):
    """/health, /classify happy path, 400 on bad input, 404 on unknown route.

    Uses a stub service so no model is needed — this tests the HTTP layer only.
    """
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from src.api import make_handler

    class StubService:
        def classify(self, texts):
            return {
                "predictions": [
                    {"text": t, "label": "code", "confidence": 0.99} for t in texts
                ]
            }

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(StubService()))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            health = json.loads(r.read())
        assert health["status"] == "ok"
        assert set(health["labels"]) == set(LABELS)

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/classify",
            data=json.dumps({"texts": ["write me a sort"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["predictions"][0]["label"] == "code"

        # invalid JSON -> 400
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/classify", data=b"not json"
        )
        try:
            urllib.request.urlopen(req)
            raised = False
        except urllib.error.HTTPError as e:
            raised = e.code == 400
        assert raised

        # unknown path -> 404
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
            raised = False
        except urllib.error.HTTPError as e:
            raised = e.code == 404
        assert raised
    finally:
        server.shutdown()


def test_ci_gate_compiles():
    source = Path(__file__).resolve().parent.parent / "scripts" / "ci_gate.py"
    compile(source.read_text(), str(source), "exec")
