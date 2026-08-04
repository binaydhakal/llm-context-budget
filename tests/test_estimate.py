from context_budget import (
    create_char_estimator,
    estimate_tokens,
    estimate_tokens_of,
    truncate_to_tokens,
)

char_tokens = create_char_estimator(1)  # 1 token per char keeps math obvious


class TestEstimators:
    def test_default_is_four_chars_per_token_rounding_up(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("12345678") == 2
        assert estimate_tokens("12345") == 2

    def test_custom_ratios(self):
        assert create_char_estimator(2)("12345678") == 4

    def test_sums_across_strings(self):
        assert estimate_tokens_of(estimate_tokens, "1234", "5678") == 2


class TestTruncateToTokens:
    def test_returns_text_unchanged_when_it_fits(self):
        assert truncate_to_tokens("hello", 10, char_tokens) == "hello"

    def test_keeps_head_by_default(self):
        assert truncate_to_tokens("abcdefghij", 4, char_tokens) == "abcd"

    def test_keeps_tail_on_request(self):
        assert truncate_to_tokens("abcdefghij", 4, char_tokens, keep="tail") == "ghij"

    def test_empty_for_zero_budget(self):
        assert truncate_to_tokens("abc", 0, char_tokens) == ""

    def test_always_lands_at_or_under_budget(self):
        result = truncate_to_tokens("x" * 1000, 123, char_tokens)
        assert char_tokens(result) <= 123
