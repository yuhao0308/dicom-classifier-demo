from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INFERENCE_BATCH_SIZE = 64
PATCH_SIZE = 24
HALF_PATCH = PATCH_SIZE // 2
SLIDING_WINDOW_STRIDE = 12
LUNG_COVERAGE_THRESHOLD = 0.25
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceModel:
    module: Any  # nn.Module or None (mock mode)
    device: Any  # torch.device or None (mock mode)
    patch_size: int = PATCH_SIZE
    mock: bool = False


@dataclass(frozen=True, slots=True)
class SliceResult:
    """One candidate detection result.

    Multiple SliceResults can share the same slice_index (one per candidate).
    """

    slice_index: int
    score: float
    x: int  # candidate center x in original slice pixel coords
    y: int  # candidate center y in original slice pixel coords


def load_mock_model() -> InferenceModel:
    """Create a mock model that generates random scores without PyTorch."""
    LOGGER.info("mock_model_loaded")
    return InferenceModel(module=None, device=None, patch_size=PATCH_SIZE, mock=True)


@lru_cache(maxsize=4)
def load_model(model_path: Path, *, use_gpu: bool = False) -> InferenceModel:
    import torch
    from torch import nn
    from torchvision.models import resnet18

    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{resolved_path}'. Run scripts/download_model.py first."
        )

    model = _build_classifier(resnet18, nn)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    device = _resolve_device(use_gpu, torch)
    model.to(device)
    model.eval()

    return InferenceModel(module=model, device=device, patch_size=PATCH_SIZE)


def generate_candidates(
    slice_array: np.ndarray,
    *,
    stride: int = SLIDING_WINDOW_STRIDE,
    lung_coverage: float = LUNG_COVERAGE_THRESHOLD,
    patch_size: int = PATCH_SIZE,
) -> list[tuple[int, int]]:
    """Generate candidate positions via sliding window over the lung region.

    Returns a list of (center_x, center_y) tuples in pixel coordinates.

    Strategy:
      1. Create a lung mask (pixels in the parenchyma intensity range)
      2. Slide a window across the image at the given stride
      3. Keep positions where the lung mask covers at least `lung_coverage`
         fraction of the patch area
    """
    if slice_array.ndim != 2:
        raise ValueError("slice_array must be a 2D array.")

    h, w = slice_array.shape
    half = patch_size // 2
    img = slice_array.astype(np.float32)

    # Lung mask: pixels in the lung parenchyma range (not background/air, not bone)
    # In lung window (W:1500 C:-600): lung parenchyma ≈ 20-180, soft tissue ≈ 170-230
    lung_mask = (img >= 10) & (img <= 240)

    # Use integral image for fast patch-level lung coverage computation
    integral = np.cumsum(np.cumsum(lung_mask.astype(np.int32), axis=0), axis=1)
    patch_area = patch_size * patch_size
    min_lung_pixels = int(lung_coverage * patch_area)

    candidates: list[tuple[int, int]] = []

    for cy in range(half, h - half + 1, stride):
        for cx in range(half, w - half + 1, stride):
            # Patch bounds
            y0 = cy - half
            y1 = cy + half
            x0 = cx - half
            x1 = cx + half

            # Sum of lung_mask within patch via integral image
            s = integral[y1 - 1, x1 - 1]
            if y0 > 0:
                s -= integral[y0 - 1, x1 - 1]
            if x0 > 0:
                s -= integral[y1 - 1, x0 - 1]
            if y0 > 0 and x0 > 0:
                s += integral[y0 - 1, x0 - 1]

            if s >= min_lung_pixels:
                candidates.append((cx, cy))

    return candidates


def extract_patch(
    slice_array: np.ndarray,
    cx: int,
    cy: int,
    *,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Extract a patch_size x patch_size patch centered at (cx, cy).

    Zero-pads if the patch extends beyond the image boundary.
    """
    half = patch_size // 2
    h, w = slice_array.shape

    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    patch = np.zeros((patch_size, patch_size), dtype=slice_array.dtype)

    dy0 = y0 - (cy - half)
    dx0 = x0 - (cx - half)
    dy1 = dy0 + (y1 - y0)
    dx1 = dx0 + (x1 - x0)

    patch[dy0:dy1, dx0:dx1] = slice_array[y0:y1, x0:x1]
    return patch


def _predict_patches_mock(
    n_patches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate mock scores — mostly low with occasional high values."""
    # Base scores: mostly low (normal tissue)
    scores = rng.beta(0.5, 5.0, size=n_patches).astype(np.float32)
    # Randomly boost a few to simulate detections (~2% of patches)
    boost_mask = rng.random(n_patches) < 0.02
    scores[boost_mask] = rng.uniform(0.6, 0.95, size=int(boost_mask.sum())).astype(np.float32)
    return scores


def predict_patches(
    model: InferenceModel,
    patches: list[np.ndarray],
) -> np.ndarray:
    """Classify a batch of patches, returning P(nodule) scores.

    Returns a 1D numpy array of shape (N,) with float32 scores in [0, 1].
    """
    if not patches:
        return np.array([], dtype=np.float32)

    if model.mock:
        rng = np.random.default_rng()
        return _predict_patches_mock(len(patches), rng)

    import torch
    from torch import Tensor

    tensors: list[Tensor] = []
    for patch in patches:
        img = np.asarray(patch, dtype=np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).expand(3, -1, -1).contiguous()
        tensors.append(t)

    batch = torch.stack(tensors, dim=0).to(device=model.device, dtype=torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype, device=model.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=batch.dtype, device=model.device).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    with torch.no_grad():
        logits = model.module(batch)
        scores = torch.softmax(logits, dim=1)[:, 1]

    return scores.cpu().numpy().astype(np.float32)


def run_inference(
    model: InferenceModel,
    slices: list[np.ndarray],
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> list[SliceResult]:
    """Run the full patch-based inference pipeline on a list of CT slices.

    For each slice:
      1. Generate candidate locations via sliding window
      2. Extract patches at each candidate
      3. Classify patches with the model
      4. Return SliceResult for each candidate with score > 0
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    all_results: list[SliceResult] = []

    for slice_idx, slice_array in enumerate(slices):
        candidates = generate_candidates(slice_array, patch_size=model.patch_size)
        if not candidates:
            continue

        LOGGER.debug(
            "slice_candidates",
            extra={"slice_index": slice_idx, "candidate_count": len(candidates)},
        )

        # Extract patches for all candidates in this slice
        patches = [
            extract_patch(slice_array, cx, cy, patch_size=model.patch_size)
            for cx, cy in candidates
        ]

        # Classify in batches
        all_scores: list[np.ndarray] = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            scores = predict_patches(model, batch)
            all_scores.append(scores)

        scores_array = np.concatenate(all_scores)

        for (cx, cy), score in zip(candidates, scores_array, strict=True):
            all_results.append(
                SliceResult(
                    slice_index=slice_idx,
                    score=float(score),
                    x=cx,
                    y=cy,
                )
            )

    return all_results


def _build_classifier(resnet18_fn: Any, nn: Any) -> Any:
    """Build the modified ResNet-18 for 24x24 patch classification.

    Must match the architecture in scripts/kaggle_train.py exactly:
      - conv1: 3x3/s1 (not 7x7/s2)
      - maxpool: Identity (removed)
      - fc: Linear(512, 2)
    Feature map progression: 24→24→12→6→3 → AdaptiveAvgPool → 512 → 2
    """
    model = resnet18_fn(weights=None)

    # Replace conv1: 7x7/s2 → 3x3/s1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(64)

    # Remove maxpool
    model.maxpool = nn.Identity()

    # 2-class output
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint  # Saved as raw state_dict.
    raise ValueError("Invalid checkpoint format. Expected a state_dict or {'state_dict': ...}.")


def _resolve_device(use_gpu: bool, torch: Any) -> Any:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
