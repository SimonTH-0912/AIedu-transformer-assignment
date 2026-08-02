"""Image validation and CLIP retrieval helpers for Lab 6."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


class InputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ImageItem:
    name: str
    data: bytes


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float


def validate_image(data: bytes, filename: str, max_bytes: int = 10_000_000) -> Image.Image:
    if len(data) > max_bytes:
        raise InputValidationError(f"{filename}: 이미지는 10 MB 이하여야 합니다.")
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise InputValidationError(f"{filename}: JPG, JPEG, PNG만 지원합니다.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InputValidationError(f"{filename}: 손상되었거나 읽을 수 없는 이미지입니다.") from exc


def normalize_embeddings(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise InputValidationError("크기가 0인 임베딩은 비교할 수 없습니다.")
    return values / norms


def rank_similar(query, candidates, limit: int) -> list[SearchResult]:
    matrix = normalize_embeddings(candidates)
    vector = normalize_embeddings(query)[0]
    if matrix.shape[1] != vector.shape[0]:
        raise InputValidationError("쿼리와 이미지 임베딩 차원이 다릅니다.")
    if not 1 <= limit <= len(matrix):
        raise InputValidationError("결과 수가 이미지 개수 범위를 벗어났습니다.")
    scores = matrix @ vector
    indices = np.argsort(scores)[::-1][:limit]
    return [SearchResult(int(index), float(scores[index])) for index in indices]


def load_clip(model_id: str = "openai/clip-vit-base-patch16"):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model


def _feature_tensor(output) -> torch.Tensor:
    """Return projected CLIP features across Transformers 4.x and 5.x."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise RuntimeError("지원하지 않는 CLIP 모델 출력 형식입니다.")


def encode_images(images: list[Image.Image], processor, model) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        vectors = _feature_tensor(model.get_image_features(**inputs))
    return normalize_embeddings(vectors.cpu().numpy())


def encode_text(text: str, processor, model) -> np.ndarray:
    if not text.strip():
        raise InputValidationError("검색 문장을 입력해 주세요.")
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vectors = _feature_tensor(model.get_text_features(**inputs))
    return normalize_embeddings(vectors.cpu().numpy())[0]
