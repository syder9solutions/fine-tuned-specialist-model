"""HTTP serving of the fine-tuned router. /health, POST /classify."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import LABELS
from src.inference import decode_predictions, score_labels


class RouterService:
    """Loads the adapter once; scores concurrently with a lock (CPU-bound)."""

    def __init__(self, run_name: str = "lora-r16", device: str = "cpu") -> None:
        from src.train import load_finetuned

        adapter_dir = Path("runs") / run_name / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"no adapter at {adapter_dir} — train first")
        self.model, self.tokenizer = load_finetuned(adapter_dir, device=device)
        self.device = device
        self.lock = threading.Lock()

    def classify(self, texts: List[str]) -> Dict[str, object]:
        with self.lock:
            logprobs = score_labels(
                self.model, self.tokenizer, texts, device=self.device
            )
        labels_out, confs = decode_predictions(logprobs)
        return {
            "predictions": [
                {"text": t, "label": l, "confidence": round(c, 4)}
                for t, l, c in zip(texts, labels_out, confs)
            ]
        }


def make_handler(service: RouterService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "labels": list(LABELS)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/classify":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                texts = payload.get("texts")
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, list) or not texts:
                    self._send(400, {"error": "body must be {'texts': [str, ...]}"})
                    return
                if len(texts) > 64:
                    self._send(400, {"error": "max 64 texts per request"})
                    return
                self._send(200, service.classify(texts))
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
            except Exception as exc:  # pragma: no cover
                self._send(500, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:  # quiet by default
            pass

    return Handler


def serve(run_name: str = "lora-r16", host: str = "127.0.0.1", port: int = 8098) -> None:
    service = RouterService(run_name=run_name)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"serving on http://{host}:{port} (POST /classify, GET /health)")
    server.serve_forever()
