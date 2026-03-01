from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_INFERENCE_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class DummyModel:
    model_path: Path


@dataclass(frozen=True, slots=True)
class SliceResult:
    slice_index: int
    score: float
    cam: np.ndarray


@lru_cache(maxsize=1)
def load_model(model_path: Path) -> DummyModel:
    return DummyModel(model_path=model_path)


def predict_batch(
    model: DummyModel,
    slices: list[np.ndarray],
    *,
    start_index: int = 0,
    rng: np.random.Generator | None = None,
) -> list[SliceResult]:
    del model  # Reserved for real model inference in commit #9.
    generator = rng or np.random.default_rng()
    results: list[SliceResult] = []

    for offset, slice_array in enumerate(slices):
        image = np.asarray(slice_array)
        if image.ndim != 2:
            raise ValueError("Each input slice must be a 2D array.")

        score = float(generator.random())
        cam = _synthetic_cam(image.shape[0], image.shape[1], generator)
        results.append(SliceResult(slice_index=start_index + offset, score=score, cam=cam))

    return results


def run_inference(
    model: DummyModel,
    slices: list[np.ndarray],
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    rng_seed: int | None = None,
) -> list[SliceResult]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    generator = np.random.default_rng(rng_seed)
    all_results: list[SliceResult] = []

    for start_index in range(0, len(slices), batch_size):
        batch = slices[start_index : start_index + batch_size]
        batch_results = predict_batch(
            model,
            batch,
            start_index=start_index,
            rng=generator,
        )
        all_results.extend(batch_results)

    return all_results


def _synthetic_cam(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    cam = gaussian_filter(
        rng.random((height, width), dtype=np.float32),
        sigma=max(min(height, width) / 12.0, 1.0),
        mode="nearest",
    ).astype(np.float32)

    min_val = float(np.min(cam))
    max_val = float(np.max(cam))
    if max_val <= min_val:
        return np.zeros((height, width), dtype=np.float32)

    return (cam - min_val) / (max_val - min_val)
