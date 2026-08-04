"""llm-context-budget -- fit chat history into any model's context window.

Zero dependencies, no AI inside; bring your own summarizer if you want one.
Python sibling of ``@yanib/context-budget`` (npm).
"""

from .estimate import (
    TokenEstimator,
    create_char_estimator,
    estimate_tokens,
    estimate_tokens_of,
    truncate_to_tokens,
)
from .pack import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_RESPONSE_RESERVE,
    MIN_RECENT_WINDOW,
    ContextLimits,
    Message,
    PackResult,
    Summarizer,
    apack_messages,
    content_text,
    context_limits_for,
    create_extractive_summarizer,
    is_context_overflow_error,
    pack_messages,
)

__version__ = "0.1.0"

__all__ = [
    "TokenEstimator",
    "create_char_estimator",
    "estimate_tokens",
    "estimate_tokens_of",
    "truncate_to_tokens",
    "DEFAULT_CONTEXT_TOKENS",
    "DEFAULT_RESPONSE_RESERVE",
    "MIN_RECENT_WINDOW",
    "ContextLimits",
    "Message",
    "PackResult",
    "Summarizer",
    "apack_messages",
    "content_text",
    "context_limits_for",
    "create_extractive_summarizer",
    "is_context_overflow_error",
    "pack_messages",
]
