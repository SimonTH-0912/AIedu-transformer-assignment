"""Validation and model adapters for the Lab 9 captioning app."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError


class InputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ImageItem:
    name: str
    data: bytes


@dataclass(frozen=True)
class ReferenceCaptions:
    english: list[str]
    korean: list[str]


@dataclass(frozen=True)
class GenerationOptions:
    max_length: int = 30
    num_beams: int = 3

    def __post_init__(self):
        if not 5 <= self.max_length <= 60:
            raise InputValidationError("최대 생성 길이는 5~60이어야 합니다.")
        if not 1 <= self.num_beams <= 8:
            raise InputValidationError("Beam 수는 1~8이어야 합니다.")


@dataclass(frozen=True)
class CaptionResult:
    filename: str
    english: str
    korean: str | None = None
    translation_warning: str | None = None
    reference: ReferenceCaptions | None = None


def validate_image_bytes(data: bytes, filename: str, max_bytes: int = 10_000_000) -> Image.Image:
    if len(data) > max_bytes:
        raise InputValidationError(f"{filename}: 이미지는 10 MB 이하여야 합니다.")
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise InputValidationError(f"{filename}: JPG, JPEG, PNG만 지원합니다.")
    try:
        image = Image.open(io.BytesIO(data)); image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InputValidationError(f"{filename}: 읽을 수 없는 이미지입니다.") from exc


def validate_images(items: list[ImageItem], max_files: int = 10) -> list[Image.Image]:
    if not items:
        raise InputValidationError("이미지를 하나 이상 선택해 주세요.")
    if len(items) > max_files:
        raise InputValidationError(f"이미지는 최대 {max_files}개까지 처리할 수 있습니다.")
    return [validate_image_bytes(item.data, item.name) for item in items]


def load_reference_captions(path: str | Path) -> dict[str, ReferenceCaptions]:
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError("정답 캡션 JSON을 읽을 수 없습니다.") from exc
    if not isinstance(records, list):
        raise InputValidationError("정답 캡션 JSON의 최상위 값은 배열이어야 합니다.")
    result = {}
    for record in records:
        filename = record.get("file_name") or Path(record.get("file_path", "")).name
        if filename:
            result[filename] = ReferenceCaptions(list(record.get("captions", [])), list(record.get("caption_ko", [])))
    return result


def load_blip(model_id="Salesforce/blip-image-captioning-base"):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    return BlipProcessor.from_pretrained(model_id), BlipForConditionalGeneration.from_pretrained(model_id).eval()


def generate_caption(image, processor, model, options: GenerationOptions, prompt="") -> str:
    inputs = processor(images=image, text=prompt or None, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=options.max_length, num_beams=options.num_beams)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError("모델이 빈 캡션을 생성했습니다.")
    return caption


def load_translator(model_id="Helsinki-NLP/opus-mt-tc-big-en-ko"):
    from transformers import MarianMTModel, MarianTokenizer

    return MarianTokenizer.from_pretrained(model_id), MarianMTModel.from_pretrained(model_id).eval()


def translate_caption(text, tokenizer, model) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=80)
    translated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if not translated:
        raise RuntimeError("번역 모델이 빈 결과를 생성했습니다.")
    return translated


def build_caption_result(filename, english, korean=None, warning=None, reference=None):
    return CaptionResult(filename, english, korean, warning, reference)

