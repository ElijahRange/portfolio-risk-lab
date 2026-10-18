"""FinBERT (Hugging Face ProsusAI/finbert) running locally on LinuxONE: financial sentence sentiment.

The model is downloaded once into data/models and loaded once per web worker, then reused.
Each sentence gets probabilities for positive / negative / neutral.
"""
import threading

from . import config

config.setup_hugging_face()

_lock = threading.Lock()
_turn = threading.Condition()
_foreground_waiting = [0]
_pipe = None
BATCH = 16


def _pipeline():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline

        torch.set_num_threads(config.NLP_THREADS)
        _pipe = pipeline("text-classification", model=config.FINBERT_MODEL, top_k=None, device="cpu")
    return _pipe


def warm_up() -> None:
    """Load the model ahead of the first request (called in a background thread at start-up)."""
    try:
        with _lock:
            _pipeline()
    except Exception:
        pass  # the first real request will raise a visible error instead


def score(sentences: list[str], background: bool = False) -> list[dict]:
    """[{positive, negative, neutral}] probabilities per sentence.

    Someone waiting on the page (foreground) always goes first: background work runs one small batch
    at a time and pauses whenever a foreground request is waiting.
    """
    if not sentences:
        return []
    out = []
    if background:
        for i in range(0, len(sentences), BATCH):
            with _turn:
                _turn.wait_for(lambda: _foreground_waiting[0] == 0)
            with _lock:
                out += _run(sentences[i:i + BATCH])
        return out
    with _turn:
        _foreground_waiting[0] += 1
    try:
        with _lock:
            return _run(sentences)
    finally:
        with _turn:
            _foreground_waiting[0] -= 1
            _turn.notify_all()


def _run(batch: list[str]) -> list[dict]:
    rows = _pipeline()(batch, batch_size=BATCH, truncation=True, max_length=128)
    return [{d["label"]: float(d["score"]) for d in row} for row in rows]
