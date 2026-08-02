"""Small, CPU-friendly Seq2Seq implementation for the Lab 2 app."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass

import torch
from torch import nn


SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]


class InputValidationError(ValueError):
    pass


def _decode(data: bytes, max_bytes: int = 5_000_000) -> str:
    if len(data) > max_bytes:
        raise InputValidationError("파일은 5 MB 이하여야 합니다.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError("UTF-8 파일을 사용해 주세요.") from exc


def validate_pairs(pairs: list[tuple[str, str]], max_pairs: int = 5000) -> list[tuple[str, str]]:
    cleaned = [(a.strip(), b.strip()) for a, b in pairs if a.strip() and b.strip()]
    if not cleaned:
        raise InputValidationError("유효한 병렬 문장이 없습니다.")
    if len(cleaned) > max_pairs:
        raise InputValidationError(f"문장쌍은 최대 {max_pairs:,}개까지 지원합니다.")
    return cleaned


def load_parallel_text(kor: bytes, eng: bytes) -> list[tuple[str, str]]:
    left, right = _decode(kor).splitlines(), _decode(eng).splitlines()
    if len(left) != len(right):
        raise InputValidationError("한국어와 영어 TXT의 줄 수가 같아야 합니다.")
    return validate_pairs(list(zip(left, right)))


def load_parallel_csv(data: bytes) -> list[tuple[str, str]]:
    reader = csv.DictReader(io.StringIO(_decode(data)))
    if not reader.fieldnames or not {"korean", "english"}.issubset(reader.fieldnames):
        raise InputValidationError("CSV에는 korean, english 열이 필요합니다.")
    return validate_pairs([(row.get("korean", ""), row.get("english", "")) for row in reader])


def tokenize_korean(text: str) -> list[str]:
    return [token for token in re.findall(r"[가-힣]+|[^\s가-힣]", text.lower()) if token.strip()]


def tokenize_english(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[^\w\s]", text.lower())


class Vocabulary:
    def __init__(self, tokens: list[str]):
        ordered = SPECIALS + [token for token in tokens if token not in SPECIALS]
        self.stoi = {token: idx for idx, token in enumerate(ordered)}
        self.itos = ordered

    @classmethod
    def build(cls, token_sequences, min_freq: int = 1):
        counts: dict[str, int] = {}
        for sequence in token_sequences:
            for token in sequence:
                counts[token] = counts.get(token, 0) + 1
        return cls(sorted(token for token, count in counts.items() if count >= min_freq))

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str], add_boundaries: bool = False) -> list[int]:
        ids = [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
        return [self.stoi["<sos>"], *ids, self.stoi["<eos>"]] if add_boundaries else ids

    def decode(self, ids) -> list[str]:
        ignored = {"<pad>", "<sos>", "<eos>"}
        return [self.itos[int(idx)] for idx in ids if self.itos[int(idx)] not in ignored]


def build_vocabularies(pairs):
    return (
        Vocabulary.build(tokenize_korean(src) for src, _ in pairs),
        Vocabulary.build(tokenize_english(tgt) for _, tgt in pairs),
    )


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, source):
        outputs, hidden = self.rnn(self.embedding(source))
        return outputs, hidden


class Attention(nn.Module):
    def forward(self, hidden, encoder_outputs, source_mask):
        scores = torch.bmm(encoder_outputs, hidden[-1].unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention()
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, token, hidden, encoder_outputs, source_mask):
        embedded = self.embedding(token).unsqueeze(1)
        context = self.attention(hidden, encoder_outputs, source_mask)
        decoded, hidden = self.rnn(torch.cat([embedded, context.unsqueeze(1)], dim=2), hidden)
        logits = self.output(torch.cat([decoded.squeeze(1), context], dim=1))
        return logits, hidden


@dataclass
class Seq2SeqBundle:
    encoder: Encoder
    decoder: Decoder
    source_vocab: Vocabulary
    target_vocab: Vocabulary


def _tensorize(text, vocab, tokenizer):
    return torch.tensor([vocab.encode(tokenizer(text), True)], dtype=torch.long)


def train_model(pairs, epochs=5, learning_rate=0.003, embedding_dim=64, hidden_dim=128, seed=7, progress_callback=None):
    if not 1 <= epochs <= 20:
        raise InputValidationError("에포크는 1~20이어야 합니다.")
    random.seed(seed)
    torch.manual_seed(seed)
    source_vocab, target_vocab = build_vocabularies(pairs)
    encoder = Encoder(len(source_vocab), embedding_dim, hidden_dim)
    decoder = Decoder(len(target_vocab), embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.stoi["<pad>"])
    losses = []
    for epoch in range(epochs):
        shuffled = list(pairs)
        random.shuffle(shuffled)
        total = 0.0
        for source_text, target_text in shuffled:
            source = _tensorize(source_text, source_vocab, tokenize_korean)
            target = _tensorize(target_text, target_vocab, tokenize_english)
            encoder_outputs, hidden = encoder(source)
            mask = source.ne(source_vocab.stoi["<pad>"])
            token = target[:, 0]
            loss = 0.0
            for step in range(1, target.size(1)):
                logits, hidden = decoder(token, hidden, encoder_outputs, mask)
                loss = loss + criterion(logits, target[:, step])
                token = target[:, step]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            total += float(loss.item()) / max(1, target.size(1) - 1)
        losses.append(total / len(shuffled))
        if progress_callback:
            progress_callback(epoch + 1, epochs, losses[-1])
    return Seq2SeqBundle(encoder, decoder, source_vocab, target_vocab), losses


def greedy_translate(text, bundle: Seq2SeqBundle, max_length=20):
    bundle.encoder.eval(); bundle.decoder.eval()
    with torch.no_grad():
        source = _tensorize(text, bundle.source_vocab, tokenize_korean)
        outputs, hidden = bundle.encoder(source)
        mask = source.ne(0)
        token = torch.tensor([bundle.target_vocab.stoi["<sos>"]])
        result = []
        for _ in range(max_length):
            logits, hidden = bundle.decoder(token, hidden, outputs, mask)
            token = logits.argmax(1)
            idx = int(token.item())
            if idx == bundle.target_vocab.stoi["<eos>"]:
                break
            result.append(idx)
    return bundle.target_vocab.decode(result)


def beam_translate(text, bundle: Seq2SeqBundle, beam_width=3, max_length=20):
    if beam_width < 1 or beam_width > 8:
        raise InputValidationError("Beam 폭은 1~8이어야 합니다.")
    bundle.encoder.eval(); bundle.decoder.eval()
    with torch.no_grad():
        source = _tensorize(text, bundle.source_vocab, tokenize_korean)
        outputs, initial_hidden = bundle.encoder(source)
        mask = source.ne(0)
        beams = [([bundle.target_vocab.stoi["<sos>"]], 0.0, initial_hidden)]
        eos = bundle.target_vocab.stoi["<eos>"]
        for _ in range(max_length):
            candidates = []
            for ids, score, hidden in beams:
                if ids[-1] == eos:
                    candidates.append((ids, score, hidden)); continue
                logits, next_hidden = bundle.decoder(torch.tensor([ids[-1]]), hidden, outputs, mask)
                values, indices = torch.log_softmax(logits, 1).topk(beam_width, 1)
                for value, index in zip(values[0], indices[0]):
                    candidates.append((ids + [int(index)], score + float(value), next_hidden.clone()))
            beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)[:beam_width]
            if all(ids[-1] == eos for ids, _, _ in beams):
                break
    return bundle.target_vocab.decode(beams[0][0][1:])


def training_fingerprint(pairs, settings) -> str:
    payload = json.dumps({"pairs": pairs, "settings": settings}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

