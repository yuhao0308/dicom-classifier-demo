from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import app.services.inference as inference
from app.services.inference import SliceResult, load_model, predict_batch, run_inference


def _sample_slices(count: int, *, height: int = 32, width: int = 32) -> list[np.ndarray]:
    return [np.full((height, width), fill_value=index, dtype=np.uint8) for index in range(count)]


def test_load_model_is_cached_for_same_path() -> None:
    model_a = load_model(Path("models/mock.pt"))
    model_b = load_model(Path("models/mock.pt"))

    assert model_a is model_b


def test_predict_batch_outputs_expected_shape_and_ranges() -> None:
    model = load_model(Path("models/mock.pt"))
    slices = _sample_slices(3, height=24, width=16)

    results = predict_batch(model, slices, start_index=5, rng=np.random.default_rng(7))

    assert [result.slice_index for result in results] == [5, 6, 7]
    for result, source_slice in zip(results, slices, strict=True):
        assert 0.0 <= result.score <= 1.0
        assert result.cam.shape == source_slice.shape
        assert result.cam.dtype == np.float32
        assert float(np.min(result.cam)) >= 0.0
        assert float(np.max(result.cam)) <= 1.0


def test_run_inference_is_deterministic_with_seed() -> None:
    model = load_model(Path("models/mock.pt"))
    slices = _sample_slices(5, height=20, width=20)

    first_run = run_inference(model, slices, batch_size=2, rng_seed=123)
    second_run = run_inference(model, slices, batch_size=2, rng_seed=123)

    assert [result.slice_index for result in first_run] == [0, 1, 2, 3, 4]
    assert [result.score for result in first_run] == [result.score for result in second_run]

    for first, second in zip(first_run, second_run, strict=True):
        assert np.array_equal(first.cam, second.cam)


def test_run_inference_chunks_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_predict_batch(
        model: object,
        slices: list[np.ndarray],
        *,
        start_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> list[SliceResult]:
        del model
        del rng
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

    model = load_model(Path("models/mock.pt"))
    slices = _sample_slices(10, height=8, width=8)
    results = run_inference(model, slices, batch_size=4, rng_seed=999)

    assert calls == [(0, 4), (4, 4), (8, 2)]
    assert len(results) == 10
    assert [result.slice_index for result in results] == list(range(10))


def test_run_inference_rejects_invalid_batch_size() -> None:
    model = load_model(Path("models/mock.pt"))

    with pytest.raises(ValueError, match="batch_size must be > 0."):
        run_inference(model, _sample_slices(1), batch_size=0)
