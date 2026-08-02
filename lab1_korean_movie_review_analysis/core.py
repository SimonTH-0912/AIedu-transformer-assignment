"""Core text-analysis functions for the Lab 1 Streamlit app."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


class InputValidationError(ValueError):
    """Raised when user-provided data cannot be analyzed safely."""


def decode_text(data: bytes, max_bytes: int = 4_000_000) -> str:
    if len(data) > max_bytes:
        raise InputValidationError(f"파일은 {max_bytes / 1_000_000:g} MB 이하여야 합니다.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputValidationError("UTF-8로 저장된 TXT 파일을 사용해 주세요.") from exc
    if not text.strip():
        raise InputValidationError("분석할 텍스트가 비어 있습니다.")
    return text


def clean_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        cleaned = re.sub(r"[^가-힣\s]", " ", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            sentences.append(cleaned)
    if not sentences:
        raise InputValidationError("한글로 구성된 분석 가능한 문장이 없습니다.")
    return sentences


def count_tokens(
    sentences: Iterable[str],
    tagger,
    stopwords: set[str],
    allowed_pos: set[str],
    min_length: int = 2,
    min_freq: int = 1,
) -> Counter:
    tokens: list[str] = []
    for sentence in sentences:
        for token, pos in tagger.pos(sentence, norm=True, stem=True):
            if pos in allowed_pos and len(token) >= min_length and token not in stopwords:
                tokens.append(token)
    counts = Counter(tokens)
    return Counter({token: count for token, count in counts.items() if count >= min_freq})


def top_tokens(counter: Counter, limit: int) -> list[tuple[str, int]]:
    if limit < 1:
        raise InputValidationError("표시할 토큰 수는 1 이상이어야 합니다.")
    return counter.most_common(limit)
