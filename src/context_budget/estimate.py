"""Token estimation without a tokenizer dependency.

Real tokenizers (tiktoken & friends) cost megabytes and vary per model; for
budgeting purposes the common ~4-characters-per-token heuristic is accurate
enough -- budgets should carry headroom anyway (see ``response_reserve``).
Every function accepts a custom estimator, so apps that already ship a
tokenizer can plug in exact counts::

    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    pack_messages(history, estimate_tokens=lambda t: len(enc.encode(t)))
"""

from __future__ import annotations

import math
from typing import Callable

# Estimate how many tokens a string will occupy.
TokenEstimator = Callable[[str], int]


def create_char_estimator(chars_per_token: float = 4.0) -> TokenEstimator:
    """Build a character-ratio estimator. English prose runs ~4 chars/token."""

    def estimate(text: str) -> int:
        return math.ceil(len(text) / chars_per_token) if text else 0

    return estimate


#: The default ~4-chars-per-token estimator.
estimate_tokens: TokenEstimator = create_char_estimator()


def estimate_tokens_of(estimator: TokenEstimator, *texts: str) -> int:
    """Sum estimated tokens across several strings."""
    return sum(estimator(t) for t in texts)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    estimator: TokenEstimator = estimate_tokens,
    keep: str = "head",
) -> str:
    """Truncate ``text`` to approximately ``max_tokens``, keeping head or tail.

    Works with any estimator: proportional cut, refined until it fits.
    """
    if max_tokens <= 0:
        return ""
    current = text
    tokens = estimator(current)
    # Proportional shrink converges in a few passes for any sane estimator.
    for _ in range(8):
        if tokens <= max_tokens:
            break
        ratio = max_tokens / tokens
        target_length = max(0, int(len(current) * ratio))
        current = current[:target_length] if keep == "head" else current[len(current) - target_length :]
        tokens = estimator(current)
    return current
