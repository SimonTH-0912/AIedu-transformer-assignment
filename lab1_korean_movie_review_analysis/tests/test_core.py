from collections import Counter

import pytest

from core import InputValidationError, clean_sentences, count_tokens, decode_text, top_tokens


class FakeTagger:
    def pos(self, sentence, norm=True, stem=True):
        return [("영화", "Noun"), ("정말", "Adverb"), ("좋다", "Adjective"), ("영화", "Noun")]


def test_decode_text_rejects_invalid_utf8_and_large_files():
    with pytest.raises(InputValidationError):
        decode_text(b"\xff")
    with pytest.raises(InputValidationError):
        decode_text(b"abc", max_bytes=2)


def test_decode_text_accepts_four_mb_upload_limit():
    assert len(decode_text(b"a" * 4_000_000)) == 4_000_000
    with pytest.raises(InputValidationError):
        decode_text(b"a" * 4_000_001)


def test_clean_and_count_tokens_apply_filters():
    assert clean_sentences("영화! 최고 123\n***\n배우가 좋아요") == ["영화 최고", "배우가 좋아요"]
    result = count_tokens(["ignored"], FakeTagger(), {"영화"}, {"Noun", "Adjective"}, 2, 1)
    assert result == Counter({"좋다": 1})
    assert top_tokens(Counter({"나": 1, "영화": 3}), 1) == [("영화", 3)]


def test_clean_sentences_analyzes_entire_uploaded_text():
    text = "가" * 200_001
    assert clean_sentences(text) == [text]
