import asyncio

import pytest

from context_budget import (
    DEFAULT_CONTEXT_TOKENS,
    content_text,
    context_limits_for,
    create_char_estimator,
    create_extractive_summarizer,
    is_context_overflow_error,
    apack_messages,
    pack_messages,
)

char_tokens = create_char_estimator(1)  # 1 token per char: obvious math


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def turns(count, size=40):
    return [
        msg("user" if i % 2 == 0 else "assistant", f"m{i} " + "x" * size)
        for i in range(count)
    ]


class TestContextLimits:
    def test_reserves_response_headroom_with_floor(self):
        limits = context_limits_for(4096)
        assert limits.context_budget == 3496
        assert limits.recent_window == 8
        assert context_limits_for(1200).context_budget == 1024

    def test_bigger_windows_keep_more_verbatim_turns(self):
        assert context_limits_for(8192).recent_window == 16
        assert context_limits_for(32000).recent_window == 24


class TestPackMessages:
    def test_empty_history(self):
        result = pack_messages([])
        assert result.messages == []
        assert result.summary == ""
        assert result.summarized_count == 0
        assert result.fits is True

    def test_small_history_stays_fully_verbatim(self):
        history = turns(4)
        result = pack_messages(history)
        assert result.messages == history
        assert result.dropped == []
        assert result.summary == ""

    def test_recent_window_verbatim_older_summarized(self):
        history = turns(12)
        result = pack_messages(history)
        assert result.summarized_count == 4
        assert len(result.dropped) == 4
        assert len(result.messages) == 9  # 1 system + 8 verbatim
        assert result.messages[0]["role"] == "system"
        assert "Earlier conversation summary" in result.messages[0]["content"]
        assert result.messages[1:] == history[4:]

    def test_trims_window_from_front_when_over_budget(self):
        history = [
            msg("user", "a" * 400),
            msg("assistant", "b" * 400),
            msg("user", "c" * 400),
            msg("assistant", "d" * 400),
        ]
        result = pack_messages(
            history,
            context_tokens=1700,  # budget: 1700 - 600 = 1100
            recent_window=4,
            estimate_tokens=char_tokens,
        )
        # 4 x 400 = 1600 > 1100 -> drop two, keep two (800 + summary fits)
        assert len(result.dropped) == 2
        assert result.messages[-1]["content"] == "d" * 400
        assert result.messages[-2]["content"] == "c" * 400
        assert result.fits is True
        assert result.used_tokens <= result.budget

    def test_cursor_prevents_refolding(self):
        history = turns(12)
        first = pack_messages(history)
        assert first.summarized_count == 4

        second = pack_messages(
            history, summary=first.summary, already_summarized=first.summarized_count
        )
        assert second.dropped == []
        assert second.summary == first.summary

        grown = history + turns(2)
        third = pack_messages(
            grown, summary=first.summary, already_summarized=first.summarized_count
        )
        assert third.summarized_count == 6
        assert third.dropped == history[4:6]

    def test_custom_summarizer_receives_dropped_turns(self):
        calls = []

        def summarize(prev, dropped):
            calls.append(dropped)
            return "CUSTOM SUMMARY"

        result = pack_messages(turns(12), summarize=summarize)
        assert len(calls) == 1
        assert len(calls[0]) == 4
        assert result.summary == "CUSTOM SUMMARY"
        assert "CUSTOM SUMMARY" in result.messages[0]["content"]

    def test_system_blocks_merge_into_one_system_message(self):
        result = pack_messages(
            turns(12),
            system_blocks=["You are a helpful assistant.", "Context: docs about tides."],
        )
        systems = [m for m in result.messages if m["role"] == "system"]
        assert len(systems) == 1
        assert "helpful assistant" in systems[0]["content"]
        assert "docs about tides" in systems[0]["content"]
        assert "Earlier conversation summary" in systems[0]["content"]

    def test_extra_fields_pass_through(self):
        history = [msg("user", "look at this", images=["file:///a.jpg"], id=7)]
        result = pack_messages(history)
        assert result.messages[0] == history[0]

    def test_aggressive_halves_the_window(self):
        result = pack_messages(turns(10), aggressive=True)
        assert len([m for m in result.messages if m["role"] != "system"]) == 4
        assert result.summarized_count == 6

    def test_fits_false_when_single_message_overflows(self):
        result = pack_messages(
            [msg("user", "z" * 2000)],
            context_tokens=1024,
            estimate_tokens=char_tokens,
        )
        assert result.fits is False
        assert len(result.messages) == 1  # still returned -- caller decides

    def test_final_guard_hard_truncates_summary(self):
        result = pack_messages(
            [msg("user", "old " * 200), msg("user", "w" * 900)],
            context_tokens=1624,  # budget 1024
            recent_window=1,
            estimate_tokens=char_tokens,
            summarize=lambda prev, dropped: "S" * 500,
        )
        # window 900 + summary must fit 1024 -> summary truncated to <= 124 tokens
        assert char_tokens(result.summary) <= 124
        assert result.fits is True

    def test_sync_pack_rejects_async_summarizer(self):
        async def summarize(prev, dropped):
            return "nope"

        with pytest.raises(TypeError, match="apack_messages"):
            pack_messages(turns(12), summarize=summarize)


class TestApackMessages:
    def test_awaits_async_summarizer(self):
        async def summarize(prev, dropped):
            await asyncio.sleep(0)
            return "ASYNC SUMMARY"

        result = asyncio.run(apack_messages(turns(12), summarize=summarize))
        assert result.summary == "ASYNC SUMMARY"
        assert result.summarized_count == 4

    def test_accepts_sync_summarizer_too(self):
        result = asyncio.run(
            apack_messages(turns(12), summarize=lambda prev, dropped: "SYNC OK")
        )
        assert result.summary == "SYNC OK"

    def test_matches_sync_result_with_default_summarizer(self):
        history = turns(12)
        sync_result = pack_messages(history)
        async_result = asyncio.run(apack_messages(history))
        assert async_result == sync_result


class TestContentText:
    def test_string_content_passes_through(self):
        assert content_text(msg("user", "hello")) == "hello"

    def test_multimodal_lists_contribute_text_parts(self):
        message = msg(
            "user",
            [
                {"type": "text", "text": "what is "},
                {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
                {"type": "text", "text": "this?"},
            ],
        )
        assert content_text(message) == "what is this?"

    def test_missing_or_odd_content_estimates_empty(self):
        assert content_text({"role": "user"}) == ""
        assert content_text(msg("user", None)) == ""


class TestExtractiveSummarizer:
    def test_labeled_lines_capped_at_tail(self):
        summarize = create_extractive_summarizer(
            estimator=char_tokens, per_turn_tokens=10, max_summary_tokens=60
        )
        out = summarize(
            "",
            [
                msg("user", "first question about tides and moons"),
                msg("assistant", "a very long answer that will surely be truncated"),
            ],
        )
        assert "User: " in out
        assert "Assistant: " in out
        assert char_tokens(out) <= 60


class TestOverflowHeuristic:
    def test_matches_overflow_shaped_messages_only(self):
        assert is_context_overflow_error(Exception("maximum context length exceeded"))
        assert is_context_overflow_error(Exception("prompt too long for window"))
        assert not is_context_overflow_error(Exception("connection refused"))
        assert not is_context_overflow_error(None)
