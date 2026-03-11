from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from app.services.inference import (
    InferenceModel,
    extract_patch,
    generate_candidates,
    load_model,
    predict_patches,
    run_inference,
)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _build_test_resnet() -> nn.Module:
    """Build a ResNet-18 matching the modified architecture."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _sample_slices(count: int, *, height: int = 32, width: int = 32) -> list[np.ndarray]:
    return [np.full((height, width), fill_value=index, dtype=np.uint8) for index in range(count)]


def test_load_model_is_cached_for_same_path(tmp_path: Path) -> None:
    model_path = tmp_path / "classifier.pt"
    base_model = _build_test_resnet()
    torch.save(base_model.state_dict(), model_path)

    load_model.cache_clear()
    model_a = load_model(model_path, use_gpu=False)
    model_b = load_model(model_path, use_gpu=False)

    assert model_a is model_b


def test_load_model_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    load_model.cache_clear()
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(missing_path, use_gpu=False)


def test_predict_patches_outputs_expected_ranges() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        device=torch.device("cpu"),
        patch_size=32,
    )
    patches = [np.full((32, 32), fill_value=100, dtype=np.uint8) for _ in range(3)]

    scores = predict_patches(model, patches)

    assert scores.shape == (3,)
    assert scores.dtype == np.float32
    for score in scores:
        assert 0.0 <= score <= 1.0


def test_predict_patches_empty_returns_empty() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        device=torch.device("cpu"),
        patch_size=32,
    )

    scores = predict_patches(model, [])
    assert scores.shape == (0,)


def test_extract_patch_center() -> None:
    img = np.arange(100).reshape(10, 10).astype(np.uint8)
    patch = extract_patch(img, cx=5, cy=5, patch_size=4)

    assert patch.shape == (4, 4)
    expected = img[3:7, 3:7]
    np.testing.assert_array_equal(patch, expected)


def test_extract_patch_edge_pads_with_zeros() -> None:
    img = np.ones((10, 10), dtype=np.uint8) * 128
    patch = extract_patch(img, cx=0, cy=0, patch_size=6)

    assert patch.shape == (6, 6)
    # Top-left corner should have zeros (padding)
    assert patch[0, 0] == 0
    assert patch[2, 2] == 0  # Still padding
    assert patch[3, 3] == 128  # Actual image data


def test_generate_candidates_sliding_window_on_lung_region() -> None:
    """Sliding window should produce candidates within lung-like regions."""
    # Create an image with a lung-like region (pixel values 10-240)
    img = np.zeros((128, 128), dtype=np.uint8)
    img[20:100, 20:100] = 120  # lung-like region

    candidates = generate_candidates(img, stride=12, patch_size=24)

    assert len(candidates) > 0
    for cx, cy in candidates:
        assert isinstance(cx, int)
        assert isinstance(cy, int)
        # All candidates should be within the lung region
        assert 20 <= cx <= 100
        assert 20 <= cy <= 100


def test_generate_candidates_empty_image() -> None:
    """All-zero image has no lung tissue, so no candidates."""
    img = np.zeros((128, 128), dtype=np.uint8)
    candidates = generate_candidates(img)
    assert candidates == []


def test_generate_candidates_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        generate_candidates(np.zeros((3, 64, 64), dtype=np.uint8))


def test_generate_candidates_stride_controls_density() -> None:
    """Smaller stride should produce more candidates."""
    img = np.full((128, 128), 120, dtype=np.uint8)

    candidates_coarse = generate_candidates(img, stride=24, patch_size=24)
    candidates_fine = generate_candidates(img, stride=12, patch_size=24)

    assert len(candidates_fine) > len(candidates_coarse)


def test_generate_candidates_skips_non_lung_regions() -> None:
    """Regions outside lung intensity range should not generate candidates."""
    # Image where left half is air (0) and right half is bone (255)
    img = np.zeros((128, 128), dtype=np.uint8)
    img[:, 64:] = 255

    candidates = generate_candidates(img, stride=12, patch_size=24)

    # Neither region qualifies as lung tissue (10-240 at 25% coverage)
    assert candidates == []


def test_run_inference_rejects_invalid_batch_size() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        device=torch.device("cpu"),
        patch_size=32,
    )

    with pytest.raises(ValueError, match="batch_size must be > 0."):
        run_inference(model, _sample_slices(1), batch_size=0)


def test_run_inference_with_empty_slices_returns_empty_results() -> None:
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        device=torch.device("cpu"),
        patch_size=32,
    )

    assert run_inference(model, [], batch_size=4) == []


def test_run_inference_returns_slice_results() -> None:
    """Integration test: run_inference on a slice with lung-like content."""
    module = _TinyClassifier().eval()
    model = InferenceModel(
        module=module,
        device=torch.device("cpu"),
        patch_size=24,
    )

    # Create a slice with lung-like tissue
    img = np.full((128, 128), 120, dtype=np.uint8)

    results = run_inference(model, [img], batch_size=16)

    # Should produce results (sliding window finds candidates in lung region)
    assert len(results) > 0
    for r in results:
        assert r.slice_index == 0
        assert 0.0 <= r.score <= 1.0
        assert isinstance(r.x, int)
        assert isinstance(r.y, int)
