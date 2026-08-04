# token-budget

[![PyPI](https://img.shields.io/pypi/v/token-budget)](https://pypi.org/project/token-budget/)
[![license](https://img.shields.io/pypi/l/token-budget)](./LICENSE)

**Fit chat history into any model's context window.**

Every chat app eventually writes the same code: estimate tokens, keep recent turns verbatim, compress the old ones, reserve room for the response, and handle the overflow error anyway. This package is that code — extracted from a production on-device assistant where the context windows are small and an overflow doesn't return a 400, it crashes the runtime.

Zero dependencies. No AI inside — the summarizer is a callback you can point at your model, with a deterministic extractive fallback built in. Messages are plain dicts in the OpenAI/Anthropic shape, multimodal content included. Python sibling of [`@yanib/context-budget`](https://github.com/binaydhakal/context-budget) (npm).

```python
from token_budget import pack_messages

result = pack_messages(
    history,                                   # [{"role": ..., "content": ...}, ...]
    context_tokens=8192,                       # your model's window
    response_reserve=600,                      # headroom for the reply
    summary=conversation.summary,              # carried from the last turn
    already_summarized=conversation.cursor,    # ...so old turns aren't re-folded
    system_blocks=[persona_prompt, rag_context],
)

reply = client.chat.completions.create(model=..., messages=result.messages)
conversation.summary = result.summary          # persist for next turn
conversation.cursor = result.summarized_count
```

Async apps use `apack_messages`, which also accepts async summarizers:

```python
result = await apack_messages(
    history,
    summarize=lambda prev, dropped: my_model_summary(prev, dropped),  # sync or async
)
```

## Install

```sh
pip install token-budget
```

## How it packs

1. The most recent turns stay **verbatim** (window size adapts to the model: 8 turns at 4k context, 16 at 8k, 24 at 16k+ — or set your own).
2. Older turns are represented by a **running summary**, carried between turns via `summary` + `summarized_count` so nothing is summarized twice.
3. If the verbatim window still blows the budget, it's **trimmed from the front** and the trimmed turns are folded into the summary — one summarizer call per pack, not one per message.
4. **Final guard:** if a single message + system blocks still overflow, the summary is hard-truncated into whatever room remains. If even that can't fit, you get `fits=False` and you decide (the messages are still returned).

System blocks and the summary are merged into **one** system message — some providers accept only a single instructions block, and every other provider tolerates it.

## Token estimation

The default estimator is the ~4-chars-per-token heuristic — deliberately dependency-free (real tokenizers cost megabytes and vary per model, and budgets carry headroom anyway). Have exact counts? Plug them in:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
pack_messages(history, estimate_tokens=lambda t: len(enc.encode(t)))
```

Also exported: `create_char_estimator(ratio)`, `truncate_to_tokens(text, max, estimator, keep="head"|"tail")`, `estimate_tokens_of`, `context_limits_for`, `content_text`.

## The overflow retry

Estimates are estimates. When the provider still throws, detect it and retry aggressively — half the verbatim window:

```python
from token_budget import is_context_overflow_error, pack_messages

try:
    return call_model(result.messages)
except Exception as err:
    if not is_context_overflow_error(err):
        raise
    retry = pack_messages(history, aggressive=True, **options)
    return call_model(retry.messages)
```

## The summarizer seam

`summarize(previous_summary, dropped_turns)` may call anything:

- **Default:** `create_extractive_summarizer()` — one compact labeled line per dropped turn, capped total size keeping the tail. Deterministic, instant, offline.
- **Your model:** semantic summaries when quality matters. If your call fails, return `previous_summary` — a memory hiccup should never block a chat turn.

Extra message keys (ids, tool calls, timestamps) pass through packing untouched, and OpenAI-style multimodal list content is estimated by its text parts.

## API

```python
pack_messages(history, *, context_tokens=4096, response_reserve=600,
              recent_window=None, aggressive=False, estimate_tokens=...,
              summary="", already_summarized=0, system_blocks=(),
              summarize=None) -> PackResult

PackResult:
    messages          # one merged system message (if any) + verbatim window
    summary           # persist and pass back next turn
    summarized_count  # pass back as already_summarized next turn
    dropped           # turns folded into the summary this call
    used_tokens, budget, fits
```

Fully typed (`py.typed`), Python 3.9+.

## License

MIT © [Binaya Dhakal](https://www.dhakalbinaya.com.np)
