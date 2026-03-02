from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18

DEFAULT_INFERENCE_BATCH_SIZE = 8
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class InferenceModel:
    module: nn.Module
    target_layer: nn.Module
    device: torch.device
    input_size: int = MODEL_INPUT_SIZE


@dataclass(frozen=True, slots=True)
class SliceResult:
    slice_index: int
    score: float
    cam: np.ndarray


@dataclass(slots=True)
class _GradCamBuffers:
    activations: Tensor | None = None
    gradients: Tensor | None = None


@lru_cache(maxsize=4)
def load_model(model_path: Path, *, use_gpu: bool = False) -> InferenceModel:
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{resolved_path}'. Run scripts/download_model.py first."
        )

    model = _build_classifier()
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    device = _resolve_device(use_gpu)
    model.to(device)
    model.eval()

    return InferenceModel(
        module=model,
        target_layer=model.layer4[-1].conv2,
        device=device,
        input_size=MODEL_INPUT_SIZE,
    )


def predict_batch(
    model: InferenceModel,
    slices: list[np.ndarray],
    *,
    start_index: int = 0,
) -> list[SliceResult]:
    if not slices:
        return []

    input_batch, original_shapes = _preprocess_slices(
        slices,
        input_size=model.input_size,
        device=model.device,
    )

    module = model.module
    module.zero_grad(set_to_none=True)

    with _capture_gradcam(model.target_layer) as buffers:
        logits = module(input_batch).flatten()
        scores = torch.sigmoid(logits)
        logits.sum().backward()

    if buffers.activations is None or buffers.gradients is None:
        raise RuntimeError("Failed to capture GradCAM tensors.")

    cams = _build_gradcams(
        activations=buffers.activations,
        gradients=buffers.gradients,
        output_shapes=original_shapes,
    )
    score_values = scores.detach().cpu().numpy().astype(np.float32)

    results: list[SliceResult] = []
    for offset, (score, cam) in enumerate(zip(score_values, cams, strict=True)):
        results.append(
            SliceResult(
                slice_index=start_index + offset,
                score=float(score),
                cam=cam,
            )
        )

    return results


def run_inference(
    model: InferenceModel,
    slices: list[np.ndarray],
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> list[SliceResult]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    all_results: list[SliceResult] = []
    for start_index in range(0, len(slices), batch_size):
        batch = slices[start_index : start_index + batch_size]
        all_results.extend(
            predict_batch(
                model,
                batch,
                start_index=start_index,
            )
        )
    return all_results


def _build_classifier() -> nn.Module:
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def _extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint  # Saved as raw state_dict.
    raise ValueError("Invalid checkpoint format. Expected a state_dict or {'state_dict': ...}.")


def _resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preprocess_slices(
    slices: list[np.ndarray],
    *,
    input_size: int,
    device: torch.device,
) -> tuple[Tensor, list[tuple[int, int]]]:
    tensors: list[Tensor] = []
    output_shapes: list[tuple[int, int]] = []

    for slice_array in slices:
        image = np.asarray(slice_array, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError("Each input slice must be a 2D array.")

        height, width = image.shape
        output_shapes.append((height, width))

        tensor = torch.from_numpy(image / 255.0).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.repeat(1, 3, 1, 1).squeeze(0)
        tensors.append(tensor)

    batch = torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=batch.dtype, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std, output_shapes


@contextmanager
def _capture_gradcam(target_layer: nn.Module) -> Iterator[_GradCamBuffers]:
    buffers = _GradCamBuffers()

    def _forward_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        buffers.activations = output

    def _backward_hook(
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            buffers.gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        yield buffers
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _build_gradcams(
    *,
    activations: Tensor,
    gradients: Tensor,
    output_shapes: list[tuple[int, int]],
) -> list[np.ndarray]:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activations).sum(dim=1, keepdim=True))

    resized_cams: list[np.ndarray] = []
    for cam_tensor, (height, width) in zip(cams, output_shapes, strict=True):
        resized = F.interpolate(
            cam_tensor.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        cam = resized.squeeze().detach().cpu().numpy().astype(np.float32)
        resized_cams.append(_normalize_cam(cam))

    return resized_cams


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    min_value = float(np.min(cam))
    max_value = float(np.max(cam))
    if max_value <= min_value:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - min_value) / (max_value - min_value)).astype(np.float32)
