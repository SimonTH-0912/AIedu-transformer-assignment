import pytest

from core import InputValidationError, Vocabulary, load_parallel_csv, load_parallel_text, training_fingerprint


def test_parallel_text_rejects_different_line_counts():
    with pytest.raises(InputValidationError):
        load_parallel_text("안녕\n감사".encode(), b"hello")


def test_csv_requires_named_columns_and_vocab_has_special_tokens():
    with pytest.raises(InputValidationError):
        load_parallel_csv(b"ko,en\na,b")
    vocab = Vocabulary.build([["hello", "world"], ["hello"]])
    assert vocab.decode(vocab.encode(["hello"], add_boundaries=True)) == ["hello"]


def test_training_fingerprint_is_stable_and_sensitive():
    pairs = [("안녕", "hello")]
    first = training_fingerprint(pairs, {"epochs": 2, "seed": 7})
    assert first == training_fingerprint(pairs, {"seed": 7, "epochs": 2})
    assert first != training_fingerprint(pairs, {"epochs": 3, "seed": 7})

