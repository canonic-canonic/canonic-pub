"""Local AI-text detector on Apple Silicon (M4) via MLX — a *real* statistical detector, not an LLM judge.

Scores text perplexity under a small base LM (token log-probs from the Metal GPU). Machine-generated
text tends to have lower perplexity. We expose the continuous perplexity; the metric layer then either
picks the F1-optimal threshold (a best-case detector) or reports threshold-free ROC-AUC against the slop
label. The honest question this answers: does a statistical detector carry ANY signal separating
AI-*slop* from AI-*legit* when both are model-written? (Predicted: AUC ~ 0.5.)
"""
from __future__ import annotations
import os
import math
import functools

MODEL_REPO = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-0.5B-4bit")  # base LM, ideal for perplexity


@functools.lru_cache(maxsize=1)
def _load():
    from mlx_lm import load
    return load(MODEL_REPO)


def perplexity(text: str) -> float:
    """Mean-token perplexity of `text` under the base LM. Lower => more model-like."""
    import mlx.core as mx
    import mlx.nn as nn
    model, tok = _load()
    ids = tok.encode(text)
    if len(ids) < 2:
        return float("nan")
    x = mx.array([ids])
    logits = model(x)[0, :-1, :].astype(mx.float32)        # (T-1, V)
    targets = mx.array(ids[1:])
    logp = nn.log_softmax(logits, axis=-1)
    tok_lp = mx.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    nll = -tok_lp.mean().item()
    return math.exp(min(nll, 20.0))


if __name__ == "__main__":
    print("model:", MODEL_REPO)
    for t in ["The TCP three-way handshake establishes a connection via SYN, SYN-ACK, and ACK segments.",
              "The Holloway theorem on entropy-stable concurrency guarantees harmonic backpressure folding."]:
        print(round(perplexity(t), 2), "|", t[:60])
