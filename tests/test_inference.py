from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

import app.services.inference as inference
from app.services.inference import (
    InferenceModel,
    SliceResult,
    load_model,
    predict_batch,
    run_inference,
)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _sample_slices(count: int, *, height: int = 32, width: int = 32) -> list[np.ndarray]:
    return [np.full((height, width), fill_value=index, dtype=np.uint8) for index in range(count)]


def test_load_model_is_cached_for_same_path(tmp_path: Path) -> None:
    model_path = tmp_path / "classifier.pt"
    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 1)
    torch.save(base_model.state_dict(), model_path)

    load_model.cache_clear()
    model_a = load_model(model_path, use_gpu=False)
    model_b = load_model(model_path, use_gpu=False)

    assert model_a is model_b


def test_predict_batch_outputs_expected_shape_and_ranges() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        target_layer=module.conv,
        device=torch.device("cpu"),
        input_size=32,
    )
    slices = _sample_slices(3, height=24, width=16)

    results = predict_batch(model, slices, start_index=5)

    assert [result.slice_index for result in results] == [5, 6, 7]
    for result, source_slice in zip(results, slices, strict=True):
        assert 0.0 <= result.score <= 1.0
        assert result.cam.shape == source_slice.shape
        assert result.cam.dtype == np.float32
        assert float(np.min(result.cam)) >= 0.0
        assert float(np.max(result.cam)) <= 1.0


def test_run_inference_chunks_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_predict_batch(
        model: InferenceModel,
        slices: list[np.ndarray],
        *,
        start_index: int = 0,
    ) -> list[SliceResult]:
        del model
        calls.append((start_index, len(slices)))
        return [
            SliceResult(
                slice_index=start_index + offset,
                score=0.5,
                cam=np.zeros_like(slice_array, dtype=np.float32),
            )
            for offset, slice_array in enumerate(slices)
        ]

    monkeypatch.setattr(inference, "predict_batch", fake_predict_batch)

    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        target_layer=module.conv,
        device=torch.device("cpu"),
        input_size=32,
    )
    slices = _sample_slices(10, height=8, width=8)
    results = run_inference(model, slices, batch_size=4)

    assert calls == [(0, 4), (4, 4), (8, 2)]
    assert len(results) == 10
    assert [result.slice_index for result in results] == list(range(10))


def test_run_inference_rejects_invalid_batch_size() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        target_layer=module.conv,
        device=torch.device("cpu"),
        input_size=32,
    )

    with pytest.raises(ValueError, match="batch_size must be > 0."):
        run_inference(model, _sample_slices(1), batch_size=0)
