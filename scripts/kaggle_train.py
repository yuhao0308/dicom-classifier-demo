"""
LUNA16 Patch-Based Nodule Classifier — Kaggle Training Script (v4)
==================================================================
Trains a modified ResNet-18 binary classifier on **24x24 candidate patches**.

Key design decisions:
  - Patch-based: classifies 48x48 patches centered on candidate locations,
    NOT full CT slices.  A 10mm nodule fills ~2-4% of a 48x48 patch
    (vs 0.07% of a 224x224 full slice).
  - Modified ResNet-18: conv1 changed from 7x7/s2 to 3x3/s1, maxpool
    removed, to preserve spatial resolution for small patches.
  - Uses LUNA16 candidates.csv (551K candidates with labels) for training.
  - Uses WeightedRandomSampler for class balance.
  - Focal Loss to focus on hard examples.
  - Preprocessing matches app pipeline exactly:
      HU → lung window (W:1500 C:-600) → uint8 [0,255] → /255 → ImageNet norm

Kaggle Setup
------------
1. Accelerator  → GPU T4 ×2
2. Internet     → ON
3. Add dataset  → "Luna16" by avc0706 (34.5 GB)
   Run kaggle_prepare_data.py in the FIRST cell.
   Run kaggle_download_luna16.py in the SECOND cell.
4. Paste this script into the THIRD cell and run.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    output_dir: Path = Path("/kaggle/working")

    # ── Patch config ───────────────────────────────────────────────────────
    patch_size: int = 24
    neg_per_pos_ratio: float = 10.0

    # ── Training ──────────────────────────────────────────────────────────
    num_classes: int = 2
    batch_size: int = 128  # small patches → larger batch
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 15
    num_workers: int = 4
    seed: int = 42
    val_split: float = 0.15

    # ── Lung windowing (matches app/services/dicom_parser.py) ─────────────
    window_center: float = -600.0
    window_width: float = 1500.0

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


cfg = Config()

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.models as models  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402
from torch.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
log.info("Device: %s  |  GPUs: %d", DEVICE, NUM_GPUS)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD PRE-EXTRACTED PATCHES
# ──────────────────────────────────────────────────────────────────────────────

PREEXTRACTED_PATH = Path("/kaggle/working/preextracted.npz")

if not PREEXTRACTED_PATH.exists():
    raise FileNotFoundError(
        f"{PREEXTRACTED_PATH} not found. Run kaggle_download_luna16.py first to extract patches."
    )

log.info("=" * 70)
log.info("Loading pre-extracted patches from %s", PREEXTRACTED_PATH)
log.info("=" * 70)
data = np.load(PREEXTRACTED_PATH, allow_pickle=True)
all_images = data["images"]
all_labels = data["labels"]
n_pos = int((all_labels == 1).sum())
n_neg = int((all_labels == 0).sum())
all_series_uids = data.get("series_uids", None)
if all_series_uids is not None and len(all_series_uids) > 0:
    all_series_uids = all_series_uids.astype(str)
else:
    all_series_uids = None
log.info(
    "Loaded %d patches (pos=%d, neg=%d, %.0f MB, shape=%s, has_uids=%s)",
    len(all_labels),
    n_pos,
    n_neg,
    all_images.nbytes / 1e6,
    all_images.shape,
    all_series_uids is not None,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. PYTORCH DATASET — FROM PRE-EXTRACTED ARRAYS
# ──────────────────────────────────────────────────────────────────────────────


class PreExtractedDataset(Dataset):
    """Ultra-fast dataset: reads from pre-extracted numpy arrays in RAM."""

    def __init__(
        self,
        images: np.ndarray,  # (N, 48, 48) uint8
        labels: np.ndarray,  # (N,) int64
        *,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # uint8 → float [0, 1]  (matches inference.py: image / 255.0)
        img = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])

        # (1, 48, 48) → (3, 48, 48)
        t = torch.from_numpy(img).unsqueeze(0).expand(3, -1, -1).contiguous()

        if self.augment:
            if torch.rand(1).item() > 0.5:
                t = T.functional.horizontal_flip(t)
            if torch.rand(1).item() > 0.5:
                t = T.functional.vertical_flip(t)
            angle = float(torch.randint(-30, 31, (1,)).item())
            t = T.functional.rotate(t, angle)
            # Random brightness/contrast jitter
            t = T.functional.adjust_brightness(t, 0.8 + 0.4 * torch.rand(1).item())
            t = T.functional.adjust_contrast(t, 0.8 + 0.4 * torch.rand(1).item())
            # Gaussian noise (10% of the time — patches are small, need more reg)
            if torch.rand(1).item() < 0.10:
                t = t + 0.03 * torch.randn_like(t)
            # Random erasing / cutout (smaller scale for small patches)
            t = T.RandomErasing(p=0.15, scale=(0.02, 0.08))(t)

        # ImageNet normalisation (matches inference.py)
        t = T.functional.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t, label


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / VAL SPLIT (by scan when UIDs available, stratified otherwise)
# ──────────────────────────────────────────────────────────────────────────────

if all_series_uids is not None:
    from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_split, random_state=cfg.seed)
    train_idx, val_idx = next(splitter.split(all_images, all_labels, groups=all_series_uids))
    log.info("Split by scan ID (no data leakage)")
else:
    from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_split, random_state=cfg.seed)
    train_idx, val_idx = next(splitter.split(all_images, all_labels))
    log.info("Split stratified (no scan UIDs available)")

train_images, train_labels = all_images[train_idx], all_labels[train_idx]
val_images, val_labels = all_images[val_idx], all_labels[val_idx]

log.info(
    "Train: %d (pos=%d neg=%d)  |  Val: %d (pos=%d neg=%d)",
    len(train_labels),
    (train_labels == 1).sum(),
    (train_labels == 0).sum(),
    len(val_labels),
    (val_labels == 1).sum(),
    (val_labels == 0).sum(),
)

train_ds = PreExtractedDataset(train_images, train_labels, augment=True)
val_ds = PreExtractedDataset(val_images, val_labels, augment=False)

# WeightedRandomSampler for balanced batches (WITHOUT weighted loss)
class_counts = np.bincount(train_labels, minlength=2).astype(np.float64)
sample_weights = np.where(train_labels == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).double(),
    num_samples=len(train_ds),
    replacement=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    sampler=sampler,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL — Modified ResNet-18 for 24x24 patches
# ──────────────────────────────────────────────────────────────────────────────
# Standard ResNet-18 conv1 (7x7/s2) + maxpool (3x3/s2) reduces 24→6 immediately.
# For 24x24 patches we need to preserve spatial resolution:
#   - Replace conv1 with 3x3/s1 (keeps 24x24)
#   - Remove maxpool (nn.Identity)
#   - layer1-4 downsample as usual: 24 → 24 → 12 → 6 → 3
#   - AdaptiveAvgPool → (512, 1, 1) → fc → 2 classes
#
# ImageNet pretrained weights are loaded for layer1-4 only.
# conv1/bn1 are randomly initialized (different kernel size).


def build_model() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace conv1: 7x7/s2 → 3x3/s1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Re-init bn1 (paired with new conv1)
    model.bn1 = nn.BatchNorm2d(64)

    # Remove maxpool
    model.maxpool = nn.Identity()

    # Replace FC for 2-class output
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)

    return model


model = build_model()
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
log.info("Model: resnet18-patch48  |  Params: %.2fM", n_params)

# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────


# Focal Loss — down-weights easy negatives, focuses on hard examples.
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


criterion = FocalLoss(gamma=2.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
scaler = GradScaler("cuda")

history: dict[str, list[float]] = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_sensitivity": [],
    "val_specificity": [],
    "val_f1": [],
}
best_val_sens = 0.0
patience_counter = 0


def run_epoch(loader: DataLoader, *, train: bool) -> dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = total = tp = fp = fn = tn = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = logits.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-8)
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


log.info("=" * 70)
log.info(
    "Starting training  |  %d epochs  |  batch %d × %d GPU(s)",
    cfg.epochs,
    cfg.batch_size,
    max(NUM_GPUS, 1),
)
log.info("=" * 70)

t0 = time.time()
for epoch in range(1, cfg.epochs + 1):
    epoch_start = time.time()
    train_m = run_epoch(train_loader, train=True)
    val_m = run_epoch(val_loader, train=False)
    scheduler.step()
    epoch_sec = time.time() - epoch_start

    for key in (
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "val_sensitivity",
        "val_specificity",
        "val_f1",
    ):
        src = train_m if key.startswith("train") else val_m
        history[key].append(src[key.replace("train_", "").replace("val_", "")])

    log.info(
        "Epoch %02d/%02d [%.0fs]  t_loss=%.4f  v_loss=%.4f  "
        "acc=%.3f  sens=%.3f  spec=%.3f  F1=%.3f  "
        "(TP=%d FP=%d FN=%d TN=%d)",
        epoch,
        cfg.epochs,
        epoch_sec,
        train_m["loss"],
        val_m["loss"],
        val_m["acc"],
        val_m["sensitivity"],
        val_m["specificity"],
        val_m["f1"],
        val_m["tp"],
        val_m["fp"],
        val_m["fn"],
        val_m["tn"],
    )

    # Checkpoint on best sensitivity, gated by minimum specificity
    if val_m["sensitivity"] > best_val_sens and val_m["specificity"] >= 0.70:
        best_val_sens = val_m["sensitivity"]
        patience_counter = 0
        state = (
            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        )
        torch.save(state, cfg.output_dir / "classifier.pt")
        log.info(
            "  ↳ Saved best model (sens=%.3f, spec=%.3f, F1=%.3f)",
            val_m["sensitivity"],
            val_m["specificity"],
            val_m["f1"],
        )
    else:
        patience_counter += 1
        if patience_counter >= cfg.patience:
            log.info("Early stopping at epoch %d", epoch)
            break

elapsed = time.time() - t0
log.info("Training complete in %.1f min", elapsed / 60)

# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE TORCHSCRIPT MODEL
# ──────────────────────────────────────────────────────────────────────────────

clean_model = build_model()
clean_model.load_state_dict(torch.load(cfg.output_dir / "classifier.pt", weights_only=True))
clean_model.eval()

scripted = torch.jit.trace(clean_model, torch.randn(1, 3, cfg.patch_size, cfg.patch_size))
scripted.save(str(cfg.output_dir / "classifier_scripted.pt"))
log.info("Saved TorchScript model → classifier_scripted.pt")

# ──────────────────────────────────────────────────────────────────────────────
# 8. SAVE HISTORY
# ──────────────────────────────────────────────────────────────────────────────

with open(cfg.output_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS (sanity check)
# ──────────────────────────────────────────────────────────────────────────────


def generate_sample_predictions() -> None:
    import matplotlib.pyplot as plt

    clean_model.eval()

    # Find some positive and negative val samples
    pos_mask = val_labels == 1
    neg_mask = val_labels == 0

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Predictions (top: positive, bottom: negative)")

    for row, (mask, title) in enumerate([(pos_mask, "Nodule"), (neg_mask, "Normal")]):
        indices = np.where(mask)[0][:5]
        for col, idx in enumerate(indices):
            ds = PreExtractedDataset(val_images[idx : idx + 1], val_labels[idx : idx + 1])
            img_tensor, label = ds[0]
            with torch.no_grad():
                logits = clean_model(img_tensor.unsqueeze(0))
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb = img_tensor.permute(1, 2, 0).numpy()
            rgb = (rgb * std + mean).clip(0, 1)

            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"{title}\nP(nod)={prob:.2f}", fontsize=9)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(cfg.output_dir / "sample_predictions.png", dpi=150)
    plt.close(fig)
    log.info("Saved sample predictions → sample_predictions.png")


try:
    generate_sample_predictions()
except Exception as e:
    log.warning("Could not generate sample predictions: %s", e)

# ──────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

log.info("=" * 70)
log.info("DONE — Output files:")
for f in sorted(cfg.output_dir.iterdir()):
    if f.is_file():
        log.info("  • %-30s  %.1f MB", f.name, f.stat().st_size / 1e6)
log.info("")
log.info(
    "Best sens=%.3f  |  Best F1=%.3f  |  Best spec=%.3f",
    best_val_sens,
    max(history["val_f1"]) if history["val_f1"] else 0,
    max(history["val_specificity"]) if history["val_specificity"] else 0,
)
log.info("")
log.info("Next: download classifier.pt → models/classifier.pt → uvicorn app.main:app --reload")
log.info("=" * 70)
