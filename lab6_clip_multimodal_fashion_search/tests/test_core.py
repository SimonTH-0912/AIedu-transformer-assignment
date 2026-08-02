import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from core import InputValidationError, encode_images, encode_text, normalize_embeddings, rank_similar, validate_image


def image_bytes(fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8), "red").save(buffer, format=fmt)
    return buffer.getvalue()


def test_validate_image_converts_to_rgb_and_rejects_bad_data():
    assert validate_image(image_bytes(), "sample.png").mode == "RGB"
    with pytest.raises(InputValidationError):
        validate_image(b"not-image", "bad.png")


def test_normalize_and_rank_cosine_similarity():
    vectors = normalize_embeddings(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    ranked = rank_similar(np.array([1.0, 0.0]), np.array([[0.0, 1.0], [1.0, 0.0]]), 1)
    assert ranked[0].index == 1
    assert ranked[0].score == pytest.approx(1.0)


class FakeProcessor:
    def __call__(self, **kwargs):
        return {"pixel_values": torch.ones(1, 3)} if "images" in kwargs else {"input_ids": torch.ones(1, 2, dtype=torch.long)}


class TransformersV5Model:
    def get_image_features(self, **kwargs):
        return SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))

    def get_text_features(self, **kwargs):
        return SimpleNamespace(pooler_output=torch.tensor([[0.0, 2.0]]))


def test_encode_functions_support_transformers_v5_model_outputs():
    processor, model = FakeProcessor(), TransformersV5Model()
    image_vector = encode_images([Image.new("RGB", (2, 2))], processor, model)
    text_vector = encode_text("white shoes", processor, model)
    assert np.allclose(image_vector, [[0.6, 0.8]])
    assert np.allclose(text_vector, [0.0, 1.0])
