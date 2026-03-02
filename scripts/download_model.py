from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and save a ResNet-18 binary classifier checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/classifier.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone initialization.",
    )
    return parser.parse_args()


def build_classifier(use_pretrained: bool) -> tuple[nn.Module, str]:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    source = "ImageNet pretrained ResNet-18 backbone"

    if weights is not None:
        try:
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)
            source = "Randomly initialized ResNet-18 backbone (pretrained download unavailable)"
    else:
        model = resnet18(weights=None)
        source = "Randomly initialized ResNet-18 backbone"

    model.fc = nn.Linear(model.fc.in_features, 1)
    model.eval()
    return model, source


def main() -> None:
    args = parse_args()
    output_path: Path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, source = build_classifier(use_pretrained=not args.no_pretrained)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": "resnet18_binary",
        "source": source,
    }
    torch.save(payload, output_path)
    print(f"Saved checkpoint to {output_path} ({source}).")


if __name__ == "__main__":
    main()
