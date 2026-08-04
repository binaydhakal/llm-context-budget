"""Sliding-window + running-summary packing for chat history.

The strategy (proven in production in an on-device assistant, where context
windows are small and overflows crash the runtime, not just the request):

- The most recent turns stay verbatim.
- Everything older is represented by a running summary carried between turns.
- If the verbatim window still blows the budget, it is trimmed from the front
  and the trimmed turns are folded into the summary.
- Final guard: if a single message + system blocks still overflow, the summary
  is hard-truncated to whatever room remains.

The summarizer is a callback -- plug in your model for semantic summaries, or
use the default extractive one (deterministic, no AI, zero dependencies).

Messages are plain dicts in the OpenAI/Anthropic shape (``role`` +
``content``); extra keys (ids, timestamps, tool calls) pass through untouched,
and multimodal list-content is estimated by its text parts.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

from .estimate import TokenEstimator, estimate_tokens, truncate_to_tokens

#: A chat message: ``{"role": ..., "content": ...}`` plus any extra keys.
Message = Dict[str, Any]

#: Folds dropped turns into the running summary. Sync for ``pack_messages``;
#: ``apack_messages`` also accepts a coroutine function.
Summarizer = Callable[[str, List[Message]], Union[str, Awaitable[str]]]

#: Fallback context window when a model doesn't declare one.
DEFAULT_CONTEXT_TOKENS = 4096
#: Tokens reserved for the model's own response by default.
DEFAULT_RESPONSE_RESERVE = 600
#: Floor for how many trailing messages stay verbatim (small windows).
MIN_RECENT_WINDOW = 8

_OVERFLOW_RE = re.compile(r"context|exceed|window|too long|too large|token", re.IGNORECASE)


@dataclass(frozen=True)
class ContextLimits:
    """Prompt budget + verbatim-window size derived from a model's window."""

    context_budget: int
    recent_window: int


def context_limits_for(
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    *,
    response_reserve: int = DEFAULT_RESPONSE_RESERVE,
    recent_window: Optional[int] = None,
) -> ContextLimits:
    """Derive the prompt budget and verbatim-window size from a context window.

    Bigger windows keep more recent turns verbatim before summarization.
    """
    budget = max(1024, context_tokens - response_reserve)
    if recent_window is None:
        recent_window = 24 if context_tokens >= 16000 else 16 if context_tokens >= 8000 else MIN_RECENT_WINDOW
    return ContextLimits(context_budget=budget, recent_window=recent_window)


def content_text(message: Message) -> str:
    """The estimable text of a message.

    Strings pass through; OpenAI-style multimodal lists contribute their text
    parts; anything else estimates as empty.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _role_label(role: str) -> str:
    if role == "user":
        return "User"
    if role == "assistant":
        return "Assistant"
    return role.capitalize()


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def create_extractive_summarizer(
    *,
    max_summary_tokens: int = 512,
    per_turn_tokens: int = 60,
    estimator: TokenEstimator = estimate_tokens,
) -> Callable[[str, List[Message]], str]:
    """The default summarizer: no AI, fully deterministic.

    Each dropped turn becomes one compact transcript line (truncated to
    ``per_turn_tokens``); the combined summary is capped at
    ``max_summary_tokens``, keeping the TAIL -- recent facts matter more than
    the opening pleasantries.
    """

    def summarize(previous_summary: str, dropped: List[Message]) -> str:
        lines = [
            f"{_role_label(str(m.get('role', '')))}: "
            f"{truncate_to_tokens(_single_line(content_text(m)), per_turn_tokens, estimator)}"
            for m in dropped
        ]
        combined = "\n".join(x for x in [previous_summary, *lines] if x)
        return truncate_to_tokens(combined, max_summary_tokens, estimator, keep="tail")

    return summarize


@dataclass
class PackResult:
    """What ``pack_messages`` returns."""

    #: Messages to send: one merged system message (if any) + verbatim window.
    messages: List[Message] = field(default_factory=list)
    #: Updated running summary -- persist and pass back next turn.
    summary: str = ""
    #: How many leading history messages the summary covers -- pass back next turn.
    summarized_count: int = 0
    #: The turns folded into the summary during THIS call.
    dropped: List[Message] = field(default_factory=list)
    #: Estimated tokens of the packed prompt.
    used_tokens: int = 0
    #: The budget the prompt had to fit (context_tokens - response_reserve).
    budget: int = 0
    #: False only when system blocks + one message inherently overflow.
    fits: bool = True


def _pack_prepare(
    history: Sequence[Message],
    *,
    context_tokens: int,
    response_reserve: int,
    recent_window: Optional[int],
    aggressive: bool,
    estimator: TokenEstimator,
    summary: str,
    already_summarized: int,
    system_blocks: Sequence[str],
):
    """Shared (non-summarizing) phase: window selection + budget trimming."""
    limits = context_limits_for(
        context_tokens, response_reserve=response_reserve, recent_window=recent_window
    )
    window_size = max(2, limits.recent_window // 2) if aggressive else limits.recent_window
    blocks = [b for b in system_blocks if b]
    extra_tokens = sum(estimator(b) for b in blocks)
    start = max(0, already_summarized)

    window_start = max(start, len(history) - window_size)
    window = list(history[window_start:])

    # Trim from the front while the budget is blown (keep >= 1 message). The
    # pre-fold summary estimate is used here; the post-fold guard catches any
    # growth the fold introduces.
    def window_tokens() -> int:
        return sum(estimator(content_text(m)) for m in window)

    while len(window) > 1 and extra_tokens + estimator(summary) + window_tokens() > limits.context_budget:
        window.pop(0)
        window_start += 1

    dropped = list(history[start:window_start])
    return limits, blocks, extra_tokens, window, window_start, dropped


def _pack_finish(
    *,
    limits: ContextLimits,
    blocks: List[str],
    extra_tokens: int,
    window: List[Message],
    window_start: int,
    dropped: List[Message],
    summary: str,
    estimator: TokenEstimator,
) -> PackResult:
    """Shared final phase: hard truncation guard + merged system message."""
    window_tokens = sum(estimator(content_text(m)) for m in window)
    if extra_tokens + estimator(summary) + window_tokens > limits.context_budget:
        allowed = max(0, limits.context_budget - window_tokens - extra_tokens)
        summary = truncate_to_tokens(summary, allowed, estimator)

    # One merged system message: some providers accept only a single
    # instructions block, and every other provider tolerates it.
    out_blocks = list(blocks)
    if summary:
        out_blocks.append(f"Earlier conversation summary (for context):\n{summary}")
    messages: List[Message] = []
    if out_blocks:
        messages.append({"role": "system", "content": "\n\n".join(out_blocks)})
    messages.extend(window)

    used = extra_tokens + estimator(summary) + window_tokens
    return PackResult(
        messages=messages,
        summary=summary,
        summarized_count=window_start,
        dropped=dropped,
        used_tokens=used,
        budget=limits.context_budget,
        fits=used <= limits.context_budget,
    )


def pack_messages(
    history: Sequence[Message],
    *,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    response_reserve: int = DEFAULT_RESPONSE_RESERVE,
    recent_window: Optional[int] = None,
    aggressive: bool = False,
    estimate_tokens: TokenEstimator = estimate_tokens,
    summary: str = "",
    already_summarized: int = 0,
    system_blocks: Sequence[str] = (),
    summarize: Optional[Summarizer] = None,
) -> PackResult:
    """Pack chat history into a model's context window.

    ``history`` should contain finalized messages only (exclude the empty
    assistant placeholder about to be streamed into). Pass the previous
    result's ``summary`` and ``summarized_count`` (as ``already_summarized``)
    so turns aren't re-folded every call.

    ``aggressive=True`` halves the verbatim window -- the reactive retry path
    after a context-overflow error (see :func:`is_context_overflow_error`).
    """
    estimator = estimate_tokens
    summarizer = summarize or create_extractive_summarizer(estimator=estimator)
    limits, blocks, extra, window, window_start, dropped = _pack_prepare(
        history,
        context_tokens=context_tokens,
        response_reserve=response_reserve,
        recent_window=recent_window,
        aggressive=aggressive,
        estimator=estimator,
        summary=summary,
        already_summarized=already_summarized,
        system_blocks=system_blocks,
    )
    if dropped:
        result = summarizer(summary, dropped)
        if inspect.isawaitable(result):
            result.close()  # type: ignore[union-attr]
            raise TypeError(
                "summarize returned an awaitable; use apack_messages() for async summarizers."
            )
        summary = result
    return _pack_finish(
        limits=limits,
        blocks=blocks,
        extra_tokens=extra,
        window=window,
        window_start=window_start,
        dropped=dropped,
        summary=summary,
        estimator=estimator,
    )


async def apack_messages(
    history: Sequence[Message],
    *,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    response_reserve: int = DEFAULT_RESPONSE_RESERVE,
    recent_window: Optional[int] = None,
    aggressive: bool = False,
    estimate_tokens: TokenEstimator = estimate_tokens,
    summary: str = "",
    already_summarized: int = 0,
    system_blocks: Sequence[str] = (),
    summarize: Optional[Summarizer] = None,
) -> PackResult:
    """Async :func:`pack_messages` -- accepts sync or async summarizers."""
    estimator = estimate_tokens
    summarizer = summarize or create_extractive_summarizer(estimator=estimator)
    limits, blocks, extra, window, window_start, dropped = _pack_prepare(
        history,
        context_tokens=context_tokens,
        response_reserve=response_reserve,
        recent_window=recent_window,
        aggressive=aggressive,
        estimator=estimator,
        summary=summary,
        already_summarized=already_summarized,
        system_blocks=system_blocks,
    )
    if dropped:
        result = summarizer(summary, dropped)
        summary = await result if inspect.isawaitable(result) else result  # type: ignore[assignment]
    return _pack_finish(
        limits=limits,
        blocks=blocks,
        extra_tokens=extra,
        window=window,
        window_start=window_start,
        dropped=dropped,
        summary=summary,
        estimator=estimator,
    )


def is_context_overflow_error(err: Any) -> bool:
    """Heuristic: does this error look like a context-window overflow?"""
    message = str(err) if err is not None else ""
    return bool(_OVERFLOW_RE.search(message))
