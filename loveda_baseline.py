"""Baseline U-Net reproducible para LoveDA (Entrega 2).

Consume el indice generado en Entrega 1, nunca usa Test y escribe solamente en
``outputs/baseline_unet``. Las etiquetas 1..7 se remapean a 0..6 y el valor
oficial 0 se convierte en 255 (ignore) dentro del entrenamiento.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


RANDOM_SEED = 42
NUM_CLASSES = 7
IGNORE_INDEX_TRAIN = 255
CLASS_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
OFFICIAL_PALETTE = np.array(
    [(0, 0, 0), (255, 255, 255), (255, 0, 0), (255, 255, 0),
     (0, 0, 255), (159, 129, 183), (0, 255, 0), (255, 195, 128)],
    dtype=np.uint8,
)


@dataclass
class ExperimentConfig:
    model: str = "U-Net baseline (from scratch)"
    seed: int = RANDOM_SEED
    input_size: int = 512
    batch_size: int = 4
    max_epochs: int = 30
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    loss: str = "CrossEntropyLoss(ignore_index=255)"
    weight_decay: float = 1e-5
    early_stopping_patience: int = 7
    augmentations: tuple[str, ...] = ("random horizontal flip", "random vertical flip", "random 90-degree rotations")
    number_of_classes: int = NUM_CLASSES
    ignore_index: int = IGNORE_INDEX_TRAIN
    base_channels: int = 32
    use_class_weights: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    mixed_precision: bool = True
    execution_datetime_utc: str = ""
    device: str = ""


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_info(seed: int) -> dict[str, object]:
    available = torch.cuda.is_available()
    info = {
        "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        "cuda_available": available, "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "seed": seed,
    }
    print("\n".join(f"{key}: {value}" for key, value in info.items()))
    return info


def resolve_dataset_root(configured: str | Path | None = None) -> Path:
    candidates = [Path(configured)] if configured else []
    if os.environ.get("LOVEDA_ROOT"):
        candidates.append(Path(os.environ["LOVEDA_ROOT"]))
    candidates.append(Path("data/LoveDA"))
    for candidate in candidates:
        if candidate.exists() and any((candidate / name).exists() for name in ("Train", "Val", "Test")):
            return candidate.resolve()
        if candidate.exists():
            matches = [p for p in candidate.rglob("Train") if p.is_dir() and (p.parent / "Val").is_dir()]
            if matches:
                return matches[0].parent.resolve()
    raise FileNotFoundError("No se encontro LoveDA. Defina LOVEDA_ROOT o ubique el dataset en data/LoveDA.")


def load_index(index_path: str | Path = "outputs/dataset_index.csv") -> pd.DataFrame:
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"Falta el indice de Entrega 1: {path}")
    frame = pd.read_csv(path)
    required = {"image_id", "split", "domain", "image_path", "mask_path", *(f"class_{i}_pixel_count" for i in range(1, 8))}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset_index.csv carece de columnas requeridas: {missing}")
    return frame


def validate_index_and_files(frame: pd.DataFrame, root: Path, check_all_files: bool = True) -> None:
    splits = set(frame["split"].astype(str))
    if not {"Train", "Val", "Test"}.issubset(splits):
        raise ValueError(f"Splits incompletos en el indice: {sorted(splits)}")
    labeled = frame[frame["split"].isin(["Train", "Val"])]
    if labeled["mask_path"].isna().any():
        raise ValueError("Train/Val contiene filas sin mascara.")
    if frame.loc[frame["split"].eq("Test"), "mask_path"].notna().any():
        raise ValueError("Test posee mascaras en el indice; revisar antes de continuar.")
    metadata_checks = (
        labeled["image_valid"].astype(bool).all() if "image_valid" in frame else True,
        labeled["mask_valid"].astype(bool).all() if "mask_valid" in frame else True,
        labeled["size_match"].astype(bool).all() if "size_match" in frame else True,
        labeled["channels"].eq(3).all() if "channels" in frame else True,
    )
    if not all(metadata_checks):
        raise ValueError("El indice reporta imagenes/mascaras invalidas, canales no RGB o dimensiones discordantes.")
    if "observed_labels" in frame:
        for value in labeled["observed_labels"].dropna().astype(str):
            observed = {int(x.strip()) for x in value.split(";") if x.strip()}
            if not observed.issubset(set(range(8))):
                raise ValueError(f"Etiqueta invalida registrada: {sorted(observed)}")
    if not check_all_files:
        return
    for row in tqdm(labeled.itertuples(index=False), total=len(labeled), desc="Validacion de archivos"):
        image_path, mask_path = root / row.image_path, root / row.mask_path
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Falta imagen o mascara: {row.image_id}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            image.load(); mask_image.load()
            mask = np.asarray(mask_image)
            if image.mode != "RGB" or mask.ndim != 2 or image.size != mask_image.size:
                raise ValueError(f"RGB/mapa 2D/tamano invalido en {row.image_id}")
            unexpected = np.setdiff1d(np.unique(mask), np.arange(8))
            if unexpected.size:
                raise ValueError(f"Etiquetas invalidas {unexpected.tolist()} en {row.image_id}")


def _multilabel_objective(y: np.ndarray, chosen: np.ndarray, target_n: int) -> float:
    target = y.sum(axis=0) * (target_n / len(y))
    counts = y[chosen].sum(axis=0)
    scale = np.maximum(target, 1.0)
    return float(np.square((counts - target) / scale).sum())


def make_splits(frame: pd.DataFrame, seed: int = RANDOM_SEED, val_fraction: float = 0.2) -> pd.DataFrame:
    """Greedy reproducible: preserva dominio y aproxima presencia de 7 clases."""
    train = frame[frame["split"].eq("Train")].copy().reset_index(drop=True)
    official_val = frame[frame["split"].eq("Val")].copy()
    presence = np.column_stack([(pd.to_numeric(train[f"class_{i}_pixel_count"], errors="coerce").fillna(0) > 0).to_numpy() for i in range(1, 8)]).astype(int)
    domains = pd.get_dummies(train["domain"].str.lower()).reindex(columns=["rural", "urban"], fill_value=0).to_numpy()
    y = np.column_stack([domains, presence])
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(train), dtype=bool)
    target_n = int(round(len(train) * val_fraction))
    rarity = (y / np.maximum(y.sum(axis=0), 1)).sum(axis=1) + rng.random(len(train)) * 1e-9
    order = np.argsort(-rarity)
    for idx in order:
        if selected.sum() >= target_n:
            break
        before = _multilabel_objective(y, selected, target_n)
        candidate = selected.copy(); candidate[idx] = True
        after = _multilabel_objective(y, candidate, target_n)
        remaining_slots = target_n - int(selected.sum())
        remaining_items = len(train) - int(selected.sum())
        if after <= before or remaining_slots >= remaining_items:
            selected[idx] = True
    if selected.sum() < target_n:
        remaining = np.flatnonzero(~selected)
        rng.shuffle(remaining)
        selected[remaining[: target_n - int(selected.sum())]] = True
    train["experiment_split"] = np.where(selected, "val_internal", "train_internal")
    official_val["experiment_split"] = "official_val"
    result = pd.concat([train, official_val], ignore_index=True)
    assert not result["split"].eq("Test").any()
    assert not set(result.loc[result.experiment_split.eq("train_internal"), "image_id"]) & set(result.loc[result.experiment_split.eq("official_val"), "image_id"])
    return result


def save_split_tables(splits: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits.rename(columns={"split": "split_original"})[["image_id", "split_original", "domain", "experiment_split", "image_path", "mask_path"]].to_csv(output_dir / "splits.csv", index=False)
    totals = splits.groupby("experiment_split").size().rename("image_count")
    summary = splits.groupby(["experiment_split", "domain"]).size().rename("domain_count").reset_index()
    summary["domain_percentage"] = 100 * summary["domain_count"] / summary["experiment_split"].map(totals)
    summary["image_count"] = summary["experiment_split"].map(totals)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    rows = []
    for split_name, group in splits.groupby("experiment_split", sort=False):
        for class_id, name in enumerate(CLASS_NAMES, 1):
            present = int((pd.to_numeric(group[f"class_{class_id}_pixel_count"], errors="coerce").fillna(0) > 0).sum())
            rows.append({"experiment_split": split_name, "class_id_original": class_id, "class_name": name,
                         "images_with_class": present, "percentage_images": 100 * present / len(group)})
    presence = pd.DataFrame(rows)
    presence.to_csv(output_dir / "class_presence_by_split.csv", index=False)
    return summary, presence


class LoveDADataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, input_size: int, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.root, self.input_size, self.augment = root, input_size, augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.image_path) as im:
            image = np.asarray(im.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        with Image.open(self.root / row.mask_path) as mm:
            mask = np.asarray(mm.resize((self.input_size, self.input_size), Image.Resampling.NEAREST), dtype=np.uint8).copy()
        if self.augment:
            if random.random() < 0.5: image, mask = np.flip(image, 1), np.flip(mask, 1)
            if random.random() < 0.5: image, mask = np.flip(image, 0), np.flip(mask, 0)
            k = random.randrange(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        target = np.where(mask == 0, IGNORE_INDEX_TRAIN, mask - 1).astype(np.int64)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target))
        return image_tensor, target_tensor, str(row.image_id), str(row.domain)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(); self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__(); self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2); self.conv = DoubleConv(out_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]: x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base); self.down1 = Down(base, base * 2); self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8); self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, base * 8); self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2); self.up4 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5,x4),x3),x2),x1))


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_confusion(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    prediction = logits.argmax(1); valid = target.ne(IGNORE_INDEX_TRAIN)
    valid_count, ignored = int(valid.sum()), int((~valid).sum())
    encoded = target[valid].to(torch.int64) * NUM_CLASSES + prediction[valid].to(torch.int64)
    confusion += torch.bincount(encoded.cpu(), minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return valid_count, ignored


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    cm = confusion.double(); tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    iou_den, dice_den = tp + fp + fn, 2 * tp + fp + fn
    iou = torch.where(iou_den > 0, tp / iou_den, torch.nan); dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.nan)
    result = {"miou": float(torch.nanmean(iou)), "dice_macro": float(torch.nanmean(dice)),
              "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() else float("nan"),
              "iou": iou.tolist(), "dice": dice.tolist(), "tp": tp.long().tolist(), "fp": fp.long().tolist(), "fn": fn.long().tolist()}
    assert 0 <= result["miou"] <= 1 and 0 <= result["dice_macro"] <= 1
    return result


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, description="", collect_domains: bool = False):
    training = optimizer is not None; model.train(training); total_loss = 0.0; count = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64); valid_pixels = ignored_pixels = 0
    domain_confusions = {name: torch.zeros_like(confusion) for name in ("Urban", "Rural")} if collect_domains else {}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets, _, domains in tqdm(loader, desc=description, leave=False):
            images, targets = images.to(device), targets.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images); loss = criterion(logits, targets)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * images.size(0); count += images.size(0)
            valid, ignored = update_confusion(confusion, logits.detach(), targets); valid_pixels += valid; ignored_pixels += ignored
            if collect_domains:
                for domain_name in domain_confusions:
                    indices = [i for i, value in enumerate(domains) if str(value).lower() == domain_name.lower()]
                    if indices:
                        update_confusion(domain_confusions[domain_name], logits[indices].detach(), targets[indices])
    metrics = metrics_from_confusion(confusion); metrics.update(loss=total_loss / count, confusion=confusion, valid_pixels=valid_pixels, ignored_pixels=ignored_pixels, image_count=count)
    if collect_domains:
        metrics["domain_metrics"] = {name: metrics_from_confusion(cm) for name, cm in domain_confusions.items()}
    return metrics


def save_history_plots(history: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    specs = [("loss_curve.png", [("train_loss", "Train"), ("val_loss", "Validation interna")], "Pérdida", "Pérdida por época"),
             ("miou_curve.png", [("val_miou", "Validation interna")], "mIoU", "mIoU de validación interna"),
             ("dice_curve.png", [("val_dice_macro", "Validation interna")], "Dice macro", "Dice macro de validación interna")]
    for filename, series, ylabel, title in specs:
        fig, ax = plt.subplots(figsize=(7, 4));
        for column, label in series: ax.plot(history.epoch, history[column], marker="o", linewidth=1.5, label=label)
        ax.set(xlabel="Época", ylabel=ylabel, title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / filename, dpi=200); plt.close(fig)


def save_confusion(confusion: torch.Tensor, output_dir: Path) -> None:
    pd.DataFrame(confusion.numpy(), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv", index_label="ground_truth")
    fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(confusion.numpy(), cmap="Blues")
    ax.set_xticks(range(7), CLASS_NAMES, rotation=45, ha="right"); ax.set_yticks(range(7), CLASS_NAMES)
    ax.set(xlabel="Predicción", ylabel="Ground truth", title="Matriz de confusión — Val oficial"); fig.colorbar(image, ax=ax, label="Píxeles")
    fig.tight_layout(); fig.savefig(output_dir / "figures/confusion_matrix.png", dpi=200); plt.close(fig)


def colorize_official(mask: np.ndarray) -> np.ndarray:
    return OFFICIAL_PALETTE[np.clip(mask, 0, 7)]


def save_qualitative(model, dataset: LoveDADataset, frame: pd.DataFrame, device, output_dir: Path, seed: int, n: int = 8) -> None:
    rng = np.random.default_rng(seed); selected = []
    for domain in ("Urban", "Rural"):
        indices = np.flatnonzero(frame.domain.str.lower().eq(domain.lower()).to_numpy()); rng.shuffle(indices); selected.extend(indices[: n // 2].tolist())
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(len(frame)), selected); rng.shuffle(remaining); selected.extend(remaining[: n-len(selected)].tolist())
    qdir = output_dir / "figures/qualitative"; qdir.mkdir(parents=True, exist_ok=True); model.eval()
    legend = [Patch(facecolor=OFFICIAL_PALETTE[i]/255, label=("ignore" if i == 0 else CLASS_NAMES[i-1])) for i in range(8)]
    for number, idx in enumerate(selected, 1):
        image, target, image_id, domain = dataset[idx]
        with torch.no_grad(): pred = model(image.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy() + 1
        gt_train = target.numpy(); gt = np.where(gt_train == IGNORE_INDEX_TRAIN, 0, gt_train + 1); error = (pred != gt) & (gt != 0)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4)); axes[0].imshow(image.permute(1,2,0)); axes[1].imshow(colorize_official(gt)); axes[2].imshow(colorize_official(pred)); axes[3].imshow(error, cmap="gray", vmin=0, vmax=1)
        for ax, title in zip(axes, ["Imagen RGB", "Ground truth", "Predicción U-Net", "Error (blanco)"]): ax.set_title(title); ax.axis("off")
        fig.suptitle(f"{image_id} — {domain}"); fig.legend(handles=legend, loc="lower center", ncol=8, fontsize=7); fig.tight_layout(rect=(0,.08,1,.94)); fig.savefig(qdir / f"sample_{number:02d}.png", dpi=200); plt.close(fig)


def per_class_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({"class_id_original": range(1,8), "class_name": CLASS_NAMES, "TP": metrics["tp"], "FP": metrics["fp"], "FN": metrics["fn"], "IoU": metrics["iou"], "Dice": metrics["dice"]})


def domain_metrics_table(domain_metrics: dict[str, dict], output_dir: Path) -> pd.DataFrame:
    rows=[]
    for domain, values in domain_metrics.items():
        for cid,name in enumerate(CLASS_NAMES,1): rows.append({"Domain":domain,"class_id_original":cid,"class_name":name,"IoU":values["iou"][cid-1],"Dice":values["dice"][cid-1],"mIoU":values["miou"],"Dice_macro":values["dice_macro"]})
    table=pd.DataFrame(rows); table.to_csv(output_dir/"metrics_by_domain.csv",index=False); return table


def smoke_tests(model: nn.Module, device: torch.device, output_dir: Path | None = None) -> None:
    model.train(); x=torch.rand(2,3,64,64,device=device); y=torch.randint(0,NUM_CLASSES,(2,64,64),device=device); y[:,0,:]=IGNORE_INDEX_TRAIN
    assert x.shape[-2:]==y.shape[-2:] and set(y.unique().tolist()).issubset({*range(7),255})
    out=model(x); assert out.shape==(2,7,64,64) and out.shape[-2:]==y.shape[-2:]
    criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_TRAIN); loss=criterion(out,y); loss.backward()
    cm=torch.zeros((7,7),dtype=torch.int64); valid,ignored=update_confusion(cm,out.detach(),y); assert valid+ignored==y.numel() and int(cm.sum())==valid
    metrics=metrics_from_confusion(cm); assert 0<=metrics["miou"]<=1 and 0<=metrics["dice_macro"]<=1
    if output_dir:
        output_dir.mkdir(parents=True,exist_ok=True); path=output_dir/"_smoke_checkpoint.pth"; torch.save(model.state_dict(),path); clone=UNet(base=model.inc.block[0].out_channels).to(device); clone.load_state_dict(torch.load(path,map_location=device,weights_only=True)); path.unlink()
    model.zero_grad(set_to_none=True); print("Pruebas de humo sintéticas: OK")


def train_experiment(config: ExperimentConfig, dataset_root: str | Path | None = None) -> dict:
    seed_everything(config.seed); runtime_info(config.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device=str(device); config.execution_datetime_utc=datetime.now(timezone.utc).isoformat()
    output=Path("outputs/baseline_unet"); figures=output/"figures"; figures.mkdir(parents=True,exist_ok=True)
    root=resolve_dataset_root(dataset_root); index=load_index(); validate_index_and_files(index,root,check_all_files=True)
    splits=make_splits(index,config.seed); save_split_tables(splits,output)
    with (output/"config.json").open("w",encoding="utf-8") as f: json.dump(asdict(config),f,ensure_ascii=False,indent=2)
    frames={name:splits[splits.experiment_split.eq(name)].reset_index(drop=True) for name in ("train_internal","val_internal","official_val")}
    datasets={name:LoveDADataset(frame,root,config.input_size,augment=name=="train_internal") for name,frame in frames.items()}
    loaders={name:DataLoader(ds,batch_size=config.batch_size,shuffle=name=="train_internal",num_workers=config.num_workers,pin_memory=config.pin_memory and device.type=="cuda") for name,ds in datasets.items()}
    model=UNet(base=config.base_channels).to(device); total,trainable=parameter_counts(model)
    (output/"model_summary.txt").write_text(f"{model}\n\nTotal parameters: {total}\nTrainable parameters: {trainable}\n",encoding="utf-8")
    print(model); print(f"Parámetros totales: {total:,}; entrenables: {trainable:,}")
    smoke_tests(model,device,output); assert not splits.experiment_split.str.contains("test",case=False).any()
    criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_TRAIN); optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=config.mixed_precision and device.type=="cuda") if device.type=="cuda" else None
    history=[]; best=-float("inf"); best_epoch=0; stale=0; started=time.perf_counter()
    for epoch in range(1,config.max_epochs+1):
        epoch_start=time.perf_counter(); train_values=run_epoch(model,loaders["train_internal"],criterion,device,optimizer,scaler,f"Epoch {epoch} train")
        val_values=run_epoch(model,loaders["val_internal"],criterion,device,description=f"Epoch {epoch} val")
        row={"epoch":epoch,"train_loss":train_values["loss"],"val_loss":val_values["loss"],"val_miou":val_values["miou"],"val_dice_macro":val_values["dice_macro"],"learning_rate":optimizer.param_groups[0]["lr"],"epoch_time_seconds":time.perf_counter()-epoch_start}
        history.append(row); pd.DataFrame(history).to_csv(output/"history.csv",index=False); print(row)
        if row["val_miou"]>best:
            best=row["val_miou"]; best_epoch=epoch; stale=0; torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"config":asdict(config),"val_miou":best},output/"best_model.pth")
        else: stale+=1
        if stale>=config.early_stopping_patience: print(f"Early stopping en epoch {epoch}."); break
    training_seconds=time.perf_counter()-started; history_df=pd.DataFrame(history); save_history_plots(history_df,figures)
    checkpoint=torch.load(output/"best_model.pth",map_location=device,weights_only=False); model.load_state_dict(checkpoint["model_state_dict"]); model.eval();
    for parameter in model.parameters(): parameter.requires_grad_(False)
    official=run_epoch(model,loaders["official_val"],criterion,device,description="Official Val (evaluación final única)",collect_domains=True)
    official_json={"loss":official["loss"],"mIoU":official["miou"],"Dice_macro":official["dice_macro"],"pixel_accuracy":official["pixel_accuracy"],"image_count":official["image_count"],"valid_pixels":official["valid_pixels"],"ignored_pixels_discarded":official["ignored_pixels"]}
    (output/"official_val_metrics.json").write_text(json.dumps(official_json,indent=2),encoding="utf-8")
    per_class=per_class_table(official); per_class.to_csv(output/"official_val_per_class.csv",index=False); save_confusion(official["confusion"],output)
    domains=domain_metrics_table(official["domain_metrics"],output)
    save_qualitative(model,datasets["official_val"],frames["official_val"],device,output,config.seed)
    pd.DataFrame([{"method":"U-Net baseline","mIoU":official["miou"],"Dice_macro":official["dice_macro"],"pixel_accuracy":official["pixel_accuracy"],"official_val_loss":official["loss"],"best_epoch":best_epoch,"trainable_parameters":trainable,"input_size":config.input_size}]).to_csv(output/"report_main_results.csv",index=False)
    per_class[["class_name","IoU","Dice"]].rename(columns={"class_name":"Clase"}).to_csv(output/"report_per_class_results.csv",index=False)
    domain_report=domains.groupby("Domain",as_index=False)[["mIoU","Dice_macro"]].first(); domain_report.to_csv(output/"report_domain_results.csv",index=False)
    summary_lines=["========================================","RESUMEN REAL — BASELINE U-NET","========================================","Dataset: LoveDA",f"Modelo: {config.model}",f"Resolución utilizada: {config.input_size}x{config.input_size}",f"Train interno: {len(frames['train_internal'])} imágenes",f"Validation interno: {len(frames['val_internal'])} imágenes",f"Official Val: {len(frames['official_val'])} imágenes",f"Mejor epoch: {best_epoch}",f"Loss official Val: {official['loss']:.6f}",f"mIoU official Val: {official['miou']:.6f}",f"Dice macro official Val: {official['dice_macro']:.6f}",f"Pixel accuracy: {official['pixel_accuracy']:.6f}",f"mIoU Urban: {domain_report.loc[domain_report.Domain.eq('Urban'),'mIoU'].iloc[0]:.6f}",f"mIoU Rural: {domain_report.loc[domain_report.Domain.eq('Rural'),'mIoU'].iloc[0]:.6f}",f"Número de parámetros: {trainable}",f"Tiempo total de entrenamiento: {training_seconds:.2f} s","IoU y Dice por clase:"]
    summary_lines += [f"- {r.class_name}: IoU={r.IoU:.6f}, Dice={r.Dice:.6f}" for r in per_class.itertuples()]; summary_lines.append("========================================")
    summary="\n".join(summary_lines); print(summary); (output/"report_summary.txt").write_text(summary+"\n",encoding="utf-8")
    best_row=per_class.loc[per_class.IoU.idxmax()]; worst_row=per_class.loc[per_class.IoU.idxmin()]; urban=float(domain_report.loc[domain_report.Domain.eq("Urban"),"mIoU"].iloc[0]); rural=float(domain_report.loc[domain_report.Domain.eq("Rural"),"mIoU"].iloc[0])
    cm=official["confusion"].clone(); cm.fill_diagonal_(0); flat=torch.argsort(cm.flatten(),descending=True)[:3]; pairs=[f"{CLASS_NAMES[int(i)//7]} → {CLASS_NAMES[int(i)%7]} ({int(cm.flatten()[i]):,} píxeles)" for i in flat]
    val_after=history_df.loc[history_df.epoch.gt(best_epoch),"val_miou"]; worsening=int((val_after.diff().fillna(0)<0).sum())
    observations=[f"El mejor checkpoint correspondió a la época {best_epoch} según mIoU de validación interna ({best:.6f}).",f"En esa época, la diferencia val_loss - train_loss fue {float(history_df.loc[history_df.epoch.eq(best_epoch),'val_loss'].iloc[0]-history_df.loc[history_df.epoch.eq(best_epoch),'train_loss'].iloc[0]):.6f}.",f"Después del mejor epoch se observaron {worsening} descensos inter-época de mIoU de validación.",f"La clase {best_row.class_name} obtuvo el mayor IoU ({best_row.IoU:.6f}), mientras que {worst_row.class_name} obtuvo el menor ({worst_row.IoU:.6f}).",f"La diferencia absoluta de mIoU entre Urban y Rural fue {abs(urban-rural):.6f}.","Mayores confusiones dirigidas: "+"; ".join(pairs)+"."]
    (output/"result_observations.txt").write_text("\n".join(observations)+"\n",encoding="utf-8")
    return {"official":official_json,"best_epoch":best_epoch,"output_dir":str(output)}


def main() -> None:
    parser=argparse.ArgumentParser(description="Baseline U-Net de LoveDA")
    parser.add_argument("--smoke-only",action="store_true",help="Ejecuta pruebas sintéticas sin requerir LoveDA")
    parser.add_argument("--dataset-root",type=Path,default=None); parser.add_argument("--epochs",type=int,default=30); parser.add_argument("--batch-size",type=int,default=4); parser.add_argument("--input-size",type=int,default=512)
    args=parser.parse_args(); config=ExperimentConfig(max_epochs=args.epochs,batch_size=args.batch_size,input_size=args.input_size); seed_everything(config.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); runtime_info(config.seed)
    if args.smoke_only: smoke_tests(UNet(base=config.base_channels).to(device),device,Path("outputs/baseline_unet"))
    else: train_experiment(config,args.dataset_root)


if __name__ == "__main__": main()
