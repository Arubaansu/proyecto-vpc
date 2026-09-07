"""Herramientas reproducibles para la exploración de LoveDA (Entrega 1).

El módulo no entrena modelos ni modifica el dataset. Recorre los archivos uno a
uno, acumula estadísticas y escribe únicamente dentro de ``outputs/``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


RANDOM_SEED = 42
DATASET_ROOT = Path(os.environ.get("LOVEDA_ROOT", "data/LoveDA"))
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
IGNORE_INDEX = 0
CLASS_NAMES = {
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}
EXPECTED_LABELS = {IGNORE_INDEX, *CLASS_NAMES}

# Paleta publicada en Semantic_Segmentation/render.py del repositorio oficial.
OFFICIAL_PALETTE = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (0, 0, 255),
    5: (159, 129, 183),
    6: (0, 255, 0),
    7: (255, 195, 128),
}

ZENODO_RECORD = "5706578"
ZENODO_DOI = "10.5281/zenodo.5706578"
ZENODO_FILES = {
    "Train.zip": (4021669263, "de2b196043ed9b4af1690b3f9a7d558f"),
    "Val.zip": (2425958254, "84cae2577468ff0b5386758bb386d31d"),
    "Test.zip": (3126023212, "a489be0090465e01fb067795d24e6b47"),
    "Datasheet.pdf": (828005, "7e0a6434f78c240cbfb18afab87404aa"),
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
}
ISSUE_TYPES = [
    "corrupt_image",
    "corrupt_mask",
    "missing_mask",
    "orphan_mask",
    "size_mismatch",
    "invalid_dimensions",
    "unexpected_channels",
    "unexpected_format",
    "unexpected_label",
    "all_ignore_mask",
    "exact_duplicate",
    "cross_split_duplicate",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - requerido para verificar Zenodo
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_downloads(archive_dir: Path) -> pd.DataFrame:
    """Compara tamaño y MD5 con el registro oficial de Zenodo."""
    rows = []
    for name, (expected_size, expected_md5) in ZENODO_FILES.items():
        path = archive_dir / name
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_md5 = md5_file(path) if exists and actual_size == expected_size else None
        rows.append(
            {
                "file": name,
                "exists": exists,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "size_ok": bool(exists and actual_size == expected_size),
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "md5_ok": bool(actual_md5 == expected_md5),
            }
        )
    return pd.DataFrame(rows)


def _download_with_resume(url: str, destination: Path) -> None:
    """Descarga estándar reanudable; escribe primero un archivo .part."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if offset and not resumed:
            offset = 0
        total_header = response.headers.get("Content-Length")
        remaining = int(total_header) if total_header else None
        with partial.open(mode) as output, tqdm(
            total=(offset + remaining) if remaining is not None else None,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                progress.update(len(chunk))
    partial.replace(destination)


def download_official_loveda(
    dataset_root: Path = DATASET_ROOT,
    archive_dir: Path = Path("data/archives"),
    include_test: bool = True,
) -> pd.DataFrame:
    """Descarga, valida y extrae LoveDA desde Zenodo, sin redescargar archivos válidos."""
    names = ["Train.zip", "Val.zip"] + (["Test.zip"] if include_test else [])
    for name in names:
        path = archive_dir / name
        expected_size, expected_md5 = ZENODO_FILES[name]
        is_valid = (
            path.is_file()
            and path.stat().st_size == expected_size
            and md5_file(path) == expected_md5
        )
        if not is_valid:
            url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content"
            _download_with_resume(url, path)
        if path.stat().st_size != expected_size or md5_file(path) != expected_md5:
            raise IOError(f"La validación oficial falló para {path}")

    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        split_name = Path(name).stem
        if not (dataset_root / split_name).exists():
            with zipfile.ZipFile(archive_dir / name) as archive:
                archive.extractall(dataset_root)
    return validate_official_downloads(archive_dir)


def find_dataset_root(configured_root: Path = DATASET_ROOT) -> Path:
    """Localiza una raíz que contenga directorios de imágenes de LoveDA."""
    candidates = [configured_root]
    environment_root = os.environ.get("LOVEDA_ROOT")
    if environment_root:
        candidates.append(Path(environment_root))
    checked: list[Path] = []
    for raw in candidates:
        if not str(raw):
            continue
        candidate = raw.expanduser().resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_dir():
            for directory in candidate.rglob("*"):
                if not directory.is_dir() or "image" not in directory.name.lower():
                    continue
                relative_parts = {part.lower() for part in directory.relative_to(candidate).parts}
                has_split = bool(relative_parts & {"train", "val", "validation", "test"})
                has_domain = bool(relative_parts & {"urban", "rural"})
                if has_split and has_domain:
                    return candidate
    locations = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "No se encontró LoveDA. Descargue Train.zip, Val.zip y Test.zip desde "
        f"https://doi.org/{ZENODO_DOI}, extráigalos y defina LOVEDA_ROOT o "
        f"DATASET_ROOT. Rutas comprobadas:\n{locations}"
    )


def summarized_tree(root: Path, max_depth: int = 3, max_entries: int = 120) -> str:
    """Árbol limitado: muestra directorios y conteos sin listar miles de tiles."""
    lines = [f"{root.name}/"]
    emitted = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal emitted
        if depth > max_depth or emitted >= max_entries:
            return
        children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        directories = [p for p in children if p.is_dir()]
        files = [p for p in children if p.is_file()]
        for child in directories:
            if emitted >= max_entries:
                return
            indent = "    " * depth
            direct_files = sum(1 for p in child.iterdir() if p.is_file())
            suffix = f" ({direct_files} archivos)" if direct_files else ""
            lines.append(f"{indent}{child.name}/{suffix}")
            emitted += 1
            visit(child, depth + 1)
        if files and depth <= max_depth:
            counts = Counter((p.suffix.lower() or "[sin extensión]") for p in files)
            summary = ", ".join(f"{ext}: {count}" for ext, count in sorted(counts.items()))
            lines.append(f"{'    ' * depth}[archivos directos: {summary}]")

    visit(root, 1)
    if emitted >= max_entries:
        lines.append("... árbol truncado ...")
    return "\n".join(lines)


def _role_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts[:-1]]
    if any("mask" in part or "label" in part or "annotation" in part for part in parts):
        return "mask"
    if any("image" in part or part in {"img", "imgs"} for part in parts):
        return "image"
    return None


def _metadata_from_path(path: Path, root: Path) -> tuple[str | None, str | None]:
    relative_parts = path.relative_to(root).parts
    split = next(
        (p for p in relative_parts if p.lower() in {"train", "val", "validation", "test"}),
        None,
    )
    domain = next((p for p in relative_parts if p.lower() in {"urban", "rural"}), None)
    return split, domain


def discover_dataset_files(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Descubre archivos por la estructura real y separa imágenes, máscaras y auxiliares."""
    images: list[dict[str, Any]] = []
    masks: list[dict[str, Any]] = []
    auxiliaries: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        role = _role_from_path(path)
        split, domain = _metadata_from_path(path, root)
        entry = {"path": path, "split": split, "domain": domain}
        if role and path.suffix.lower() not in IMAGE_SUFFIXES:
            auxiliaries.append({**entry, "reason": "unsupported_extension_in_data_directory"})
        elif role == "image":
            images.append(entry)
        elif role == "mask":
            masks.append(entry)
        else:
            auxiliaries.append({**entry, "reason": "auxiliary_or_unclassified"})
    return images, masks, auxiliaries


def _relative(path: Path | None, root: Path) -> str | None:
    return path.relative_to(root).as_posix() if path is not None else None


def _issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    description: str,
    sample_id: str | None,
    split: str | None,
    domain: str | None,
    image_path: str | None,
    mask_path: str | None,
) -> None:
    issues.append(
        {
            "sample_id": sample_id,
            "issue_type": issue_type,
            "description": description,
            "split": split,
            "domain": domain,
            "image_path": image_path,
            "mask_path": mask_path,
        }
    )


def _inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_valid": False,
        "width": None,
        "height": None,
        "channels": None,
        "image_mode": None,
        "image_format": None,
        "image_file_size_bytes": path.stat().st_size,
        "sha256_image": None,
        "error": None,
    }
    try:
        result["sha256_image"] = sha256_file(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update(
                {
                    "image_valid": True,
                    "width": image.width,
                    "height": image.height,
                    "channels": len(image.getbands()),
                    "image_mode": image.mode,
                    "image_format": image.format,
                }
            )
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _inspect_mask(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mask_valid": False,
        "mask_width": None,
        "mask_height": None,
        "mask_mode": None,
        "mask_format": None,
        "mask_file_size_bytes": path.stat().st_size,
        "sha256_mask": None,
        "label_counts": {},
        "mask_total_pixels": None,
        "error": None,
    }
    try:
        result["sha256_mask"] = sha256_file(path)
        with Image.open(path) as mask_image:
            mask_image.load()
            result.update(
                {
                    "mask_width": mask_image.width,
                    "mask_height": mask_image.height,
                    "mask_mode": mask_image.mode,
                    "mask_format": mask_image.format,
                }
            )
            mask = np.asarray(mask_image)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            result["error"] = f"La máscara no es un mapa entero 2D (shape={mask.shape}, dtype={mask.dtype})"
            return result
        flat = mask.reshape(-1).astype(np.int64, copy=False)
        if flat.size and flat.min() >= 0 and flat.max() <= 65535:
            counts = np.bincount(flat)
            result["label_counts"] = {
                int(value): int(count)
                for value, count in enumerate(counts)
                if count > 0
            }
        else:
            values, counts = np.unique(flat, return_counts=True)
            result["label_counts"] = {int(v): int(c) for v, c in zip(values, counts)}
        result["mask_total_pixels"] = int(flat.size)
        result["mask_valid"] = True
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_dataset_index(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye una fila por imagen y registra controles de calidad reales."""
    image_entries, mask_entries, auxiliary_entries = discover_dataset_files(root)
    issues: list[dict[str, Any]] = []

    mask_lookup: dict[tuple[Any, ...], list[Path]] = {}
    for entry in mask_entries:
        key = (
            (entry["split"] or "").lower(),
            (entry["domain"] or "").lower(),
            entry["path"].stem.lower(),
        )
        mask_lookup.setdefault(key, []).append(entry["path"])

    used_masks: set[Path] = set()
    rows: list[dict[str, Any]] = []
    ordered_images = sorted(image_entries, key=lambda x: str(x["path"]).lower())
    for entry in tqdm(ordered_images, desc="Inspección de imágenes y máscaras", mininterval=1.0):
        image_path: Path = entry["path"]
        split, domain = entry["split"], entry["domain"]
        sample_id = "/".join(filter(None, [split, domain, image_path.stem]))
        key = ((split or "").lower(), (domain or "").lower(), image_path.stem.lower())
        candidates = sorted(mask_lookup.get(key, []))
        mask_path = candidates[0] if len(candidates) == 1 else None
        if len(candidates) > 1:
            _issue(
                issues,
                "unexpected_format",
                f"Emparejamiento ambiguo: {len(candidates)} máscaras con el mismo identificador.",
                sample_id,
                split,
                domain,
                _relative(image_path, root),
                None,
            )
        if mask_path:
            used_masks.add(mask_path)

        image_info = _inspect_image(image_path)
        relative_image = _relative(image_path, root)
        relative_mask = _relative(mask_path, root)
        if not image_info["image_valid"]:
            _issue(issues, "corrupt_image", image_info["error"], sample_id, split, domain, relative_image, relative_mask)
        elif (image_info["width"] or 0) <= 0 or (image_info["height"] or 0) <= 0:
            _issue(issues, "invalid_dimensions", "Dimensiones de imagen no positivas.", sample_id, split, domain, relative_image, relative_mask)
        if image_info["image_valid"] and image_info["channels"] != 3:
            _issue(issues, "unexpected_channels", f"Se observaron {image_info['channels']} canales; se esperaban 3 para RGB.", sample_id, split, domain, relative_image, relative_mask)
        expected_image_format = FORMAT_BY_SUFFIX.get(image_path.suffix.lower())
        if image_info["image_valid"] and expected_image_format and image_info["image_format"] != expected_image_format:
            _issue(issues, "unexpected_format", f"Extensión {image_path.suffix.lower()} pero Pillow detectó {image_info['image_format']}.", sample_id, split, domain, relative_image, relative_mask)

        mask_info: dict[str, Any] = {
            "mask_valid": False,
            "mask_width": None,
            "mask_height": None,
            "mask_mode": None,
            "mask_format": None,
            "mask_file_size_bytes": None,
            "sha256_mask": None,
            "label_counts": {},
            "mask_total_pixels": None,
            "error": None,
        }
        expected_unlabeled = (split or "").lower() == "test"
        if mask_path is None and not expected_unlabeled:
            _issue(issues, "missing_mask", "No se encontró una máscara local emparejada.", sample_id, split, domain, relative_image, None)
        else:
            if mask_path is not None:
                mask_info = _inspect_mask(mask_path)
                if not mask_info["mask_valid"]:
                    _issue(issues, "corrupt_mask", mask_info["error"], sample_id, split, domain, relative_image, relative_mask)
                expected_mask_format = FORMAT_BY_SUFFIX.get(mask_path.suffix.lower())
                if mask_info["mask_valid"] and expected_mask_format and mask_info["mask_format"] != expected_mask_format:
                    _issue(issues, "unexpected_format", f"Extensión de máscara {mask_path.suffix.lower()} pero Pillow detectó {mask_info['mask_format']}.", sample_id, split, domain, relative_image, relative_mask)
                if mask_info["mask_valid"] and ((mask_info["mask_width"] or 0) <= 0 or (mask_info["mask_height"] or 0) <= 0):
                    _issue(issues, "invalid_dimensions", "Dimensiones de máscara no positivas.", sample_id, split, domain, relative_image, relative_mask)

        label_counts = mask_info["label_counts"]
        observed = set(label_counts)
        unexpected = sorted(observed - EXPECTED_LABELS)
        if unexpected:
            _issue(issues, "unexpected_label", f"Etiquetas fuera de 0-7: {unexpected}", sample_id, split, domain, relative_image, relative_mask)
        if mask_info["mask_valid"] and label_counts and sum(label_counts.get(i, 0) for i in CLASS_NAMES) == 0:
            _issue(issues, "all_ignore_mask", "La máscara contiene únicamente no-data/ignore u otras etiquetas no válidas.", sample_id, split, domain, relative_image, relative_mask)

        size_match = None
        if image_info["image_valid"] and mask_info["mask_valid"]:
            size_match = bool(
                image_info["width"] == mask_info["mask_width"]
                and image_info["height"] == mask_info["mask_height"]
            )
            if not size_match:
                _issue(issues, "size_mismatch", f"Imagen {image_info['width']}x{image_info['height']} vs máscara {mask_info['mask_width']}x{mask_info['mask_height']}.", sample_id, split, domain, relative_image, relative_mask)

        total_pixels = mask_info["mask_total_pixels"] or 0
        ignore_count = int(label_counts.get(IGNORE_INDEX, 0))
        valid_count = int(sum(label_counts.get(i, 0) for i in CLASS_NAMES))
        row = {
            "image_id": sample_id,
            "split": split,
            "domain": domain,
            "image_path": relative_image,
            "mask_path": relative_mask,
            "has_mask": mask_path is not None,
            "width": image_info["width"],
            "height": image_info["height"],
            "channels": image_info["channels"],
            "image_mode": image_info["image_mode"],
            "image_format": image_info["image_format"],
            "image_extension": image_path.suffix.lower(),
            "mask_mode": mask_info["mask_mode"],
            "mask_format": mask_info["mask_format"],
            "mask_extension": mask_path.suffix.lower() if mask_path else None,
            "image_file_size_bytes": image_info["image_file_size_bytes"],
            "mask_file_size_bytes": mask_info["mask_file_size_bytes"],
            "image_valid": image_info["image_valid"],
            "mask_valid": mask_info["mask_valid"],
            "size_match": size_match,
            "mask_total_pixels": total_pixels if mask_info["mask_valid"] else None,
            "valid_pixel_count": valid_count if mask_info["mask_valid"] else None,
            "num_valid_classes_present": sum(label_counts.get(i, 0) > 0 for i in CLASS_NAMES) if mask_info["mask_valid"] else None,
            "ignore_pixel_count": ignore_count if mask_info["mask_valid"] else None,
            "ignore_ratio": ignore_count / total_pixels if total_pixels else None,
            "observed_labels": ";".join(map(str, sorted(observed))) if observed else None,
            "sha256_image": image_info["sha256_image"],
            "sha256_mask": mask_info["sha256_mask"],
        }
        for class_id in CLASS_NAMES:
            row[f"class_{class_id}_pixel_count"] = int(label_counts.get(class_id, 0)) if mask_info["mask_valid"] else None
        rows.append(row)

    for entry in mask_entries:
        if entry["path"] not in used_masks:
            split, domain = entry["split"], entry["domain"]
            sample_id = "/".join(filter(None, [split, domain, entry["path"].stem]))
            orphan_relative = _relative(entry["path"], root)
            _issue(issues, "orphan_mask", "Máscara sin imagen emparejada.", sample_id, split, domain, None, orphan_relative)
            orphan_info = _inspect_mask(entry["path"])
            if not orphan_info["mask_valid"]:
                _issue(issues, "corrupt_mask", orphan_info["error"], sample_id, split, domain, None, orphan_relative)
            unexpected = sorted(set(orphan_info["label_counts"]) - EXPECTED_LABELS)
            if unexpected:
                _issue(issues, "unexpected_label", f"Etiquetas fuera de 0-7: {unexpected}", sample_id, split, domain, None, orphan_relative)

    for entry in auxiliary_entries:
        if entry["reason"] == "unsupported_extension_in_data_directory":
            _issue(issues, "unexpected_format", f"Archivo no compatible en directorio de datos: {entry['path'].suffix or '[sin extensión]'}", None, entry["split"], entry["domain"], _relative(entry["path"], root), None)

    index_df = pd.DataFrame(rows)
    issue_columns = ["sample_id", "issue_type", "description", "split", "domain", "image_path", "mask_path"]
    issues_df = pd.DataFrame(issues, columns=issue_columns)
    auxiliary_df = pd.DataFrame(auxiliary_entries)
    return index_df, issues_df, auxiliary_df


def detect_duplicates(index_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa igualdad exacta byte a byte de imágenes mediante SHA-256."""
    columns = [
        "duplicate_group_id",
        "image_id",
        "split",
        "domain",
        "image_path",
        "mask_path",
        "sha256_image",
        "sha256_mask",
        "group_size",
        "within_split_duplicate",
        "cross_split_duplicate",
    ]
    valid = index_df[index_df["sha256_image"].notna()].copy()
    repeated_hashes = valid.loc[valid.duplicated("sha256_image", keep=False), "sha256_image"].unique()
    rows: list[dict[str, Any]] = []
    for number, digest in enumerate(sorted(repeated_hashes), start=1):
        group = valid[valid["sha256_image"] == digest]
        split_counts = group["split"].fillna("No disponible").value_counts()
        within = bool((split_counts > 1).any())
        cross = group["split"].dropna().nunique() > 1
        for _, sample in group.iterrows():
            rows.append(
                {
                    "duplicate_group_id": f"DUP-{number:04d}",
                    "image_id": sample["image_id"],
                    "split": sample["split"],
                    "domain": sample["domain"],
                    "image_path": sample["image_path"],
                    "mask_path": sample["mask_path"],
                    "sha256_image": digest,
                    "sha256_mask": sample["sha256_mask"],
                    "group_size": len(group),
                    "within_split_duplicate": within,
                    "cross_split_duplicate": cross,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def add_duplicate_flags(index_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> pd.DataFrame:
    result = index_df.copy()
    result["duplicate_group_id"] = None
    result["cross_split_duplicate"] = False
    if duplicates_df.empty:
        return result
    mapping = duplicates_df.set_index("image_id")
    result["duplicate_group_id"] = result["image_id"].map(mapping["duplicate_group_id"])
    result["cross_split_duplicate"] = result["image_id"].map(mapping["cross_split_duplicate"]).fillna(False).astype(bool)
    return result


def add_duplicate_issues(issues_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> pd.DataFrame:
    if duplicates_df.empty:
        return issues_df
    new_rows = []
    for _, row in duplicates_df.iterrows():
        issue_type = "cross_split_duplicate" if row["cross_split_duplicate"] else "exact_duplicate"
        new_rows.append(
            {
                "sample_id": row["image_id"],
                "issue_type": issue_type,
                "description": f"Imagen idéntica byte a byte en {row['duplicate_group_id']} (n={row['group_size']}).",
                "split": row["split"],
                "domain": row["domain"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
            }
        )
    return pd.concat([issues_df, pd.DataFrame(new_rows)], ignore_index=True)


def _readable_masks(index_df: pd.DataFrame) -> pd.DataFrame:
    return index_df[index_df["mask_valid"].fillna(False)].copy()


def make_class_tables(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = _readable_masks(index_df)
    total_pixels = int(masks["mask_total_pixels"].fillna(0).sum())
    total_valid = int(masks["valid_pixel_count"].fillna(0).sum())
    n_masks = len(masks)
    pixel_rows = []
    presence_rows = []
    for class_id, class_name in CLASS_NAMES.items():
        counts = masks[f"class_{class_id}_pixel_count"].fillna(0).astype(np.int64)
        present = counts > 0
        coverage = (counts[present] / masks.loc[present, "valid_pixel_count"].replace(0, np.nan)) * 100
        pixel_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": int(counts.sum()),
                "percentage_all_pixels": 100 * counts.sum() / total_pixels if total_pixels else np.nan,
                "percentage_valid_pixels": 100 * counts.sum() / total_valid if total_valid else np.nan,
                "images_present": int(present.sum()),
                "percentage_images_present": 100 * present.sum() / n_masks if n_masks else np.nan,
                "mean_coverage_when_present_pct_valid": coverage.mean(),
                "median_coverage_when_present_pct_valid": coverage.median(),
                "min_coverage_when_present_pct_valid": coverage.min(),
                "max_coverage_when_present_pct_valid": coverage.max(),
            }
        )
        presence_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "images_present": int(present.sum()),
                "images_with_readable_mask": n_masks,
                "percentage_images_present": 100 * present.sum() / n_masks if n_masks else np.nan,
            }
        )
    return pd.DataFrame(pixel_rows), pd.DataFrame(presence_rows)


def _group_distribution(index_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    masks = _readable_masks(index_df)
    rows = []
    for group, frame in masks.groupby(group_column, dropna=False):
        total = int(frame["mask_total_pixels"].fillna(0).sum())
        valid = int(frame["valid_pixel_count"].fillna(0).sum())
        for label_id in range(8):
            if label_id == IGNORE_INDEX:
                count = int(frame["ignore_pixel_count"].fillna(0).sum())
                name = "no-data / ignore"
                present = int((frame["ignore_pixel_count"].fillna(0) > 0).sum())
            else:
                count = int(frame[f"class_{label_id}_pixel_count"].fillna(0).sum())
                name = CLASS_NAMES[label_id]
                present = int((frame[f"class_{label_id}_pixel_count"].fillna(0) > 0).sum())
            rows.append(
                {
                    group_column: group if pd.notna(group) else "No disponible",
                    "label_id": label_id,
                    "label_name": name,
                    "pixel_count": count,
                    "percentage_all_pixels": 100 * count / total if total else np.nan,
                    "percentage_valid_pixels": (100 * count / valid if valid and label_id != IGNORE_INDEX else np.nan),
                    "images_present": present,
                    "images_with_readable_mask": len(frame),
                    "percentage_images_present": 100 * present / len(frame) if len(frame) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_group_tables(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_images = len(index_df)
    domain_rows = []
    for domain, frame in index_df.groupby("domain", dropna=False):
        masks = _readable_masks(frame)
        pixels = int(masks["mask_total_pixels"].fillna(0).sum())
        ignore = int(masks["ignore_pixel_count"].fillna(0).sum())
        domain_rows.append(
            {
                "domain": domain if pd.notna(domain) else "No disponible",
                "image_count": len(frame),
                "percentage_images": 100 * len(frame) / total_images if total_images else np.nan,
                "mask_count": int(frame["has_mask"].sum()),
                "readable_mask_count": len(masks),
                "ignore_pixel_count": ignore,
                "ignore_percentage_all_pixels": 100 * ignore / pixels if pixels else np.nan,
            }
        )

    split_rows = []
    for split, frame in index_df.groupby("split", dropna=False):
        masks = _readable_masks(frame)
        pixels = int(masks["mask_total_pixels"].fillna(0).sum())
        ignore = int(masks["ignore_pixel_count"].fillna(0).sum())
        domain_counts = frame["domain"].fillna("No disponible").value_counts()
        split_rows.append(
            {
                "split": split if pd.notna(split) else "No disponible",
                "image_count": len(frame),
                "percentage_images": 100 * len(frame) / total_images if total_images else np.nan,
                "mask_count": int(frame["has_mask"].sum()),
                "readable_mask_count": len(masks),
                "urban_image_count": int(domain_counts.get("Urban", domain_counts.get("urban", 0))),
                "rural_image_count": int(domain_counts.get("Rural", domain_counts.get("rural", 0))),
                "ignore_pixel_count": ignore,
                "ignore_percentage_all_pixels": 100 * ignore / pixels if pixels else np.nan,
            }
        )
    return (
        pd.DataFrame(domain_rows),
        pd.DataFrame(split_rows),
        _group_distribution(index_df, "domain"),
        _group_distribution(index_df, "split"),
    )


def make_summary_table(
    index_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
) -> pd.DataFrame:
    masks = _readable_masks(index_df)
    total_pixels = int(masks["mask_total_pixels"].fillna(0).sum())
    valid_pixels = int(masks["valid_pixel_count"].fillna(0).sum())
    ignore_pixels = int(masks["ignore_pixel_count"].fillna(0).sum())
    observed = sorted(
        {
            int(label)
            for labels in masks["observed_labels"].dropna()
            for label in str(labels).split(";")
            if label != ""
        }
    )
    resolutions = index_df.dropna(subset=["width", "height"]).apply(
        lambda row: f"{int(row['width'])}x{int(row['height'])}", axis=1
    )
    most_common_resolution = resolutions.mode().iloc[0] if not resolutions.empty else "No disponible"
    exact_groups = int(duplicates_df["duplicate_group_id"].nunique()) if not duplicates_df.empty else 0
    cross_groups = int(duplicates_df.loc[duplicates_df["cross_split_duplicate"], "duplicate_group_id"].nunique()) if not duplicates_df.empty else 0
    test = index_df[index_df["split"].astype(str).str.lower() == "test"]
    test_without_local_ground_truth = int((~test["has_mask"]).sum())
    metrics: list[tuple[str, Any]] = [
        ("official_source", f"Zenodo DOI {ZENODO_DOI}"),
        ("dataset", "LoveDA"),
        ("image_count", len(index_df)),
        ("mask_count", int(index_df["has_mask"].sum())),
        ("readable_mask_count", len(masks)),
        ("splits_found", "; ".join(map(str, sorted(index_df["split"].dropna().unique())))),
        ("domains_found", "; ".join(map(str, sorted(index_df["domain"].dropna().unique())))),
        ("test_has_local_masks", bool(index_df.loc[index_df["split"].astype(str).str.lower() == "test", "has_mask"].any())),
        ("test_without_local_ground_truth_count", test_without_local_ground_truth),
        ("image_formats", "; ".join(map(str, sorted(index_df["image_format"].dropna().unique())))),
        ("mask_formats", "; ".join(map(str, sorted(index_df["mask_format"].dropna().unique())))),
        ("resolutions", "; ".join(sorted(resolutions.unique())) if not resolutions.empty else "No disponible"),
        ("most_common_resolution", most_common_resolution),
        ("channels", "; ".join(map(lambda x: str(int(x)), sorted(index_df["channels"].dropna().unique())))),
        ("observed_mask_labels", "; ".join(map(str, observed)) if observed else "No disponible"),
        ("ignore_index", IGNORE_INDEX),
        ("total_mask_pixels", total_pixels),
        ("valid_pixels_1_to_7", valid_pixels),
        ("ignore_pixels", ignore_pixels),
        ("ignore_percentage_all_pixels", 100 * ignore_pixels / total_pixels if total_pixels else np.nan),
        ("observed_valid_class_count", len(set(observed) & set(CLASS_NAMES))),
        ("mean_valid_classes_per_image", masks["num_valid_classes_present"].mean()),
        ("median_valid_classes_per_image", masks["num_valid_classes_present"].median()),
        ("min_valid_classes_per_image", masks["num_valid_classes_present"].min()),
        ("max_valid_classes_per_image", masks["num_valid_classes_present"].max()),
        ("issue_count", len(issues_df)),
        ("exact_duplicate_groups", exact_groups),
        ("cross_split_duplicate_groups", cross_groups),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _open_rgb(root: Path, relative_path: str) -> np.ndarray:
    with Image.open(root / relative_path) as image:
        return np.asarray(image.convert("RGB"))


def _open_mask(root: Path, relative_path: str) -> np.ndarray:
    with Image.open(root / relative_path) as image:
        image.load()
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"Máscara no bidimensional: {relative_path}")
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, rgb in OFFICIAL_PALETTE.items():
        color[mask == label] = rgb
    unexpected = ~np.isin(mask, list(OFFICIAL_PALETTE))
    color[unexpected] = (255, 0, 255)
    return color


def _diverse_sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected: list[int] = []
    for _, group in frame.groupby(["split", "domain"], dropna=False, sort=True):
        available = group.index.to_numpy()
        selected.append(int(rng.choice(available)))
        if len(selected) >= n:
            break
    if len(selected) < min(n, len(frame)):
        remaining = frame.index.difference(selected).to_numpy()
        needed = min(n - len(selected), len(remaining))
        if needed:
            selected.extend(map(int, rng.choice(remaining, size=needed, replace=False)))
    return frame.loc[selected]


def figure_random_images(index_df: pd.DataFrame, root: Path, path: Path, rng: np.random.Generator) -> bool:
    frame = index_df[index_df["image_valid"]].copy()
    chosen = _diverse_sample(frame, 6, rng)
    if chosen.empty:
        return False
    cols = 3
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for axis, (_, sample) in zip(axes.flat, chosen.iterrows()):
        axis.imshow(_open_rgb(root, sample["image_path"]))
        axis.set_title(f"{sample['image_id']}\n{sample['domain']} · {sample['split']}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(chosen):]:
        axis.axis("off")
    fig.suptitle("LoveDA: muestra RGB aleatoria reproducible", fontsize=14)
    _save_figure(fig, path)
    return True


def figure_ground_truth_overlay(index_df: pd.DataFrame, root: Path, path: Path, rng: np.random.Generator) -> bool:
    frame = index_df[index_df["image_valid"] & index_df["mask_valid"] & index_df["size_match"].eq(True)]
    chosen = _diverse_sample(frame, 4, rng)
    if chosen.empty:
        return False
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 4 * len(chosen)), squeeze=False)
    for row_index, (_, sample) in enumerate(chosen.iterrows()):
        image = _open_rgb(root, sample["image_path"])
        mask = _open_mask(root, sample["mask_path"])
        colored = _colorize_mask(mask)
        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(colored)
        axes[row_index, 2].imshow(image)
        axes[row_index, 2].imshow(colored, alpha=0.45)
        axes[row_index, 0].set_ylabel(f"{sample['domain']} · {sample['split']}\n{sample['image_id']}", fontsize=8)
        for axis in axes[row_index]:
            axis.axis("off")
    for axis, title in zip(axes[0], ["Imagen RGB", "Ground truth", "Overlay"]):
        axis.set_title(title)
    legend = [Patch(facecolor=np.array(OFFICIAL_PALETTE[0]) / 255, label="no-data / ignore")]
    legend += [Patch(facecolor=np.array(OFFICIAL_PALETTE[i]) / 255, label=name) for i, name in CLASS_NAMES.items()]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=True)
    fig.suptitle("LoveDA: RGB, etiquetas originales y superposición", fontsize=14)
    fig.subplots_adjust(bottom=0.08)
    _save_figure(fig, path)
    return True


def figure_split_distribution(split_df: pd.DataFrame, path: Path) -> bool:
    if split_df.empty:
        return False
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(split_df["split"].astype(str), split_df["image_count"], color="#4472C4")
    axis.bar_label(bars, padding=3)
    axis.set(title="Cantidad de imágenes por split", xlabel="Split encontrado", ylabel="Imágenes")
    _save_figure(fig, path)
    return True


def figure_domain_distribution(domain_df: pd.DataFrame, path: Path) -> bool:
    if domain_df.empty:
        return False
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(domain_df["domain"].astype(str), domain_df["image_count"], color=["#70AD47", "#ED7D31"][: len(domain_df)])
    axis.bar_label(bars, padding=3)
    axis.set(title="Cantidad de imágenes por dominio", xlabel="Dominio encontrado", ylabel="Imágenes")
    _save_figure(fig, path)
    return True


def figure_class_pixels(class_df: pd.DataFrame, path: Path) -> bool:
    if class_df.empty or class_df["pixel_count"].sum() == 0:
        return False
    x = np.arange(len(class_df))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x - width / 2, class_df["percentage_valid_pixels"], width, label="% de píxeles válidos (1–7)", color="#4472C4")
    axis.bar(x + width / 2, class_df["percentage_all_pixels"], width, label="% de todos los píxeles", color="#A5A5A5")
    axis.set_xticks(x, class_df["class_name"], rotation=25, ha="right")
    axis.set(title="Distribución de píxeles por clase", xlabel="Clase", ylabel="Porcentaje (%)")
    axis.legend()
    _save_figure(fig, path)
    return True


def figure_class_presence(presence_df: pd.DataFrame, path: Path) -> bool:
    if presence_df.empty:
        return False
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(presence_df["class_name"], presence_df["percentage_images_present"], color="#5B9BD5")
    axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    axis.set(title="Presencia de clases en imágenes con máscara legible", xlabel="Clase", ylabel="Imágenes donde aparece (%)", ylim=(0, 108))
    axis.tick_params(axis="x", rotation=25)
    _save_figure(fig, path)
    return True


def figure_urban_rural(distribution_df: pd.DataFrame, path: Path) -> bool:
    frame = distribution_df[distribution_df["label_id"].between(1, 7)].copy()
    if frame.empty or frame["domain"].nunique() < 2:
        return False
    pivot = frame.pivot(index="label_name", columns="domain", values="percentage_valid_pixels")
    pivot = pivot.reindex([CLASS_NAMES[i] for i in CLASS_NAMES])
    fig, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=axis, color=["#70AD47", "#ED7D31"][: len(pivot.columns)])
    axis.set(title="Distribución de clases: Urban vs Rural", xlabel="Clase", ylabel="Píxeles válidos del dominio (%)")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Dominio")
    _save_figure(fig, path)
    return True


def figure_classes_per_image(index_df: pd.DataFrame, path: Path) -> bool:
    values = _readable_masks(index_df)["num_valid_classes_present"].dropna().astype(int)
    if values.empty:
        return False
    counts = values.value_counts().sort_index().reindex(range(0, 8), fill_value=0)
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(counts.index, counts.values, color="#8064A2")
    axis.bar_label(bars, padding=3)
    axis.set(title="Número de clases válidas distintas por imagen", xlabel="Clases presentes (ignore=0 excluido)", ylabel="Imágenes", xticks=range(0, 8))
    _save_figure(fig, path)
    return True


def figure_ignore_distribution(index_df: pd.DataFrame, path: Path) -> bool:
    values = 100 * _readable_masks(index_df)["ignore_ratio"].dropna()
    if values.empty or float(values.sum()) == 0:
        return False
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(values, bins=30, color="#7F7F7F", edgecolor="white")
    axis.set(title="Distribución del porcentaje no-data por imagen", xlabel="Píxeles ignore=0 por imagen (%)", ylabel="Imágenes")
    _save_figure(fig, path)
    return True


def _representative_selection(index_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    frame = _readable_masks(index_df)
    frame = frame[frame["image_valid"] & frame["size_match"].eq(True)].copy()
    if frame.empty:
        return []
    candidates: list[tuple[str, int]] = [
        ("Mayor diversidad de clases", int(frame["num_valid_classes_present"].idxmax())),
        ("Menor diversidad de clases", int(frame["num_valid_classes_present"].idxmin())),
    ]
    for class_id, label in [(4, "Mayor cobertura de water"), (2, "Mayor cobertura de building"), (6, "Mayor cobertura de forest"), (7, "Mayor cobertura de agriculture")]:
        ratios = frame[f"class_{class_id}_pixel_count"] / frame["valid_pixel_count"].replace(0, np.nan)
        if ratios.notna().any():
            candidates.append((label, int(ratios.idxmax())))
    selected = []
    used: set[str] = set()
    for criterion, index in candidates:
        sample = frame.loc[index]
        if sample["image_id"] not in used:
            selected.append((criterion, sample))
            used.add(sample["image_id"])
    return selected


def figure_representative_examples(index_df: pd.DataFrame, root: Path, path: Path) -> bool:
    selected = _representative_selection(index_df)
    if not selected:
        return False
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 4 * len(selected)), squeeze=False)
    for row_index, (criterion, sample) in enumerate(selected):
        image = _open_rgb(root, sample["image_path"])
        mask = _open_mask(root, sample["mask_path"])
        colored = _colorize_mask(mask)
        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(image)
        axes[row_index, 1].imshow(colored, alpha=0.45)
        axes[row_index, 0].set_title(f"{criterion}\n{sample['image_id']}", fontsize=9)
        axes[row_index, 1].set_title(f"Overlay · {sample['domain']} · {sample['split']}", fontsize=9)
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    fig.suptitle("Ejemplos representativos seleccionados automáticamente", fontsize=14)
    _save_figure(fig, path)
    return True


def generate_figures(
    index_df: pd.DataFrame,
    class_df: pd.DataFrame,
    presence_df: pd.DataFrame,
    split_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    domain_distribution_df: pd.DataFrame,
    root: Path,
    figures_dir: Path,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    jobs = [
        ("01_random_images.png", lambda p: figure_random_images(index_df, root, p, rng)),
        ("02_ground_truth_overlay.png", lambda p: figure_ground_truth_overlay(index_df, root, p, rng)),
        ("03_split_distribution.png", lambda p: figure_split_distribution(split_df, p)),
        ("04_domain_distribution.png", lambda p: figure_domain_distribution(domain_df, p)),
        ("05_class_pixel_distribution.png", lambda p: figure_class_pixels(class_df, p)),
        ("06_class_image_presence.png", lambda p: figure_class_presence(presence_df, p)),
        ("07_urban_vs_rural_classes.png", lambda p: figure_urban_rural(domain_distribution_df, p)),
        ("08_classes_per_image.png", lambda p: figure_classes_per_image(index_df, p)),
        ("09_ignore_distribution.png", lambda p: figure_ignore_distribution(index_df, p)),
        ("10_representative_examples.png", lambda p: figure_representative_examples(index_df, root, p)),
    ]
    records = []
    for filename, function in jobs:
        created = function(figures_dir / filename)
        reason = None
        if not created:
            reason = "No había datos reales suficientes; no se creó una figura vacía."
        records.append({"figure": filename, "created": created, "reason_if_omitted": reason})
    return pd.DataFrame(records)


def validate_analysis(
    index_df: pd.DataFrame,
    class_df: pd.DataFrame,
    domain_distribution_df: pd.DataFrame,
    split_distribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Comprobaciones internas sobre emparejamientos, píxeles y porcentajes."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    record("image_id_unique", index_df["image_id"].is_unique, "Una fila por identificador de imagen.")
    record("ignore_index_is_zero", IGNORE_INDEX == 0, "El valor ignore no fue remapeado.")
    readable = _readable_masks(index_df)
    accounted = readable["ignore_pixel_count"].fillna(0)
    for class_id in CLASS_NAMES:
        accounted = accounted + readable[f"class_{class_id}_pixel_count"].fillna(0)
    observed_expected_only = readable["observed_labels"].dropna().apply(
        lambda text: set(map(int, str(text).split(";"))).issubset(EXPECTED_LABELS)
    ).all()
    if observed_expected_only:
        pixels_ok = bool((accounted.astype(np.int64) == readable["mask_total_pixels"].astype(np.int64)).all())
        pixel_detail = "Ignore más clases 1-7 coincide con el total de cada máscara."
    else:
        pixels_ok = bool((accounted <= readable["mask_total_pixels"]).all())
        pixel_detail = "Hay etiquetas inesperadas; los conteos conocidos no exceden el total."
    record("pixel_accounting", pixels_ok, pixel_detail)
    valid_sum = class_df["percentage_valid_pixels"].sum()
    record("valid_class_percentages_sum_100", bool(np.isclose(valid_sum, 100.0, atol=1e-8)), f"Suma={valid_sum:.12f}%")
    record("readable_masks_have_dimensions", bool((readable["mask_total_pixels"] > 0).all()), "Todas las máscaras legibles tienen píxeles.")
    record("rgb_input_channels_checked", bool(index_df.loc[index_df["image_valid"], "channels"].notna().all()), "Los canales se obtuvieron desde cada archivo legible.")

    for group_name, table, column in [
        ("domain", domain_distribution_df, "domain"),
        ("split", split_distribution_df, "split"),
    ]:
        valid_rows = table[table["label_id"].between(1, 7)]
        sums = valid_rows.groupby(column)["percentage_valid_pixels"].sum()
        passed = bool(sums.dropna().apply(lambda value: np.isclose(value, 100.0, atol=1e-8)).all())
        record(f"{group_name}_valid_percentages_sum_100", passed, "; ".join(f"{key}={value:.9f}%" for key, value in sums.items()))
    return pd.DataFrame(checks)


def experimental_design_text(index_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> str:
    split_names = list(map(str, sorted(index_df["split"].dropna().unique())))
    test = index_df[index_df["split"].astype(str).str.lower() == "test"]
    test_has_masks = bool(test["has_mask"].any()) if not test.empty else False
    labeled_splits = [
        str(split)
        for split, frame in index_df.groupby("split", dropna=False)
        if frame["has_mask"].any()
    ]
    duplicate_note = (
        f"Se detectaron {duplicates_df['duplicate_group_id'].nunique()} grupos exactos; todo grupo debe asignarse como unidad."
        if not duplicates_df.empty
        else "No se detectaron duplicados exactos por SHA-256, aunque este hash no detecta vecindad geográfica ni similitud visual."
    )
    if "Train" in split_names and any(name.lower() in {"val", "validation"} for name in split_names) and not test_has_masks:
        strategy = (
            "Conservar Test oficial completamente intacto para la evaluación externa; sin su ground truth distribuido localmente "
            "no se pueden calcular mIoU ni Dice locales sobre este split. Usar Train oficial como datos de desarrollo y, en una "
            "entrega posterior, dividirlo en 80% de entrenamiento interno y 20% de validación interna. Mantener Val oficial "
            "completamente separado como conjunto de evaluación local final."
        )
    else:
        strategy = (
            "Usar únicamente splits con máscaras locales para desarrollo y reservar un subconjunto etiquetado independiente; "
            "la decisión final debe respetar la estructura observada."
        )
    return (
        f"Splits observados: {', '.join(split_names) or 'No disponible'}. "
        f"Splits con alguna máscara local: {', '.join(labeled_splits) or 'ninguno'}. "
        f"Test tiene máscaras locales: {'sí' if test_has_masks else 'no'}. {strategy} "
        "La futura subdivisión será por índices (sin mover/copiar archivos), usará RANDOM_SEED, preservará aproximadamente la "
        "proporción Urban/Rural, considerará la distribución multietiqueta por presencia/cobertura de clases y mantendrá juntos "
        "los eventuales duplicados exactos. No se materializa ningún split en esta entrega. "
        f"{duplicate_note}"
    )


def print_automatic_summary(
    index_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    class_df: pd.DataFrame,
) -> None:
    masks = _readable_masks(index_df)
    splits = index_df.groupby("split", dropna=False).size()
    domains = index_df.groupby("domain", dropna=False).size()
    observed = sorted(
        {
            int(label)
            for text in masks["observed_labels"].dropna()
            for label in str(text).split(";")
            if label
        }
    )
    resolutions = index_df.dropna(subset=["width", "height"]).apply(lambda r: f"{int(r.width)}x{int(r.height)}", axis=1)
    issue_counts = issues_df["issue_type"].value_counts() if not issues_df.empty else pd.Series(dtype=int)
    exact_groups = duplicates_df["duplicate_group_id"].nunique() if not duplicates_df.empty else 0
    cross_groups = duplicates_df.loc[duplicates_df["cross_split_duplicate"], "duplicate_group_id"].nunique() if not duplicates_df.empty else 0
    total_pixels = int(masks["mask_total_pixels"].fillna(0).sum())
    ignore_pixels = int(masks["ignore_pixel_count"].fillna(0).sum())

    print("RESUMEN AUTOMÁTICO PARA EL INFORME")
    print(f"Fuente oficial: Zenodo DOI {ZENODO_DOI}")
    print("Dataset: LoveDA")
    print(f"Número de imágenes: {len(index_df)}")
    print(f"Número de máscaras: {int(index_df['has_mask'].sum())}")
    print("Splits encontrados:")
    for split, count in splits.items():
        print(f"- {split if pd.notna(split) else 'No disponible'}: {count}")
    test = index_df[index_df["split"].astype(str).str.lower() == "test"]
    test_has_masks = bool(test["has_mask"].any())
    test_without_local_ground_truth = int((~test["has_mask"]).sum())
    print(f"Test tiene máscaras locales: {'sí' if test_has_masks else 'no'}")
    print(f"Test sin ground truth local: {test_without_local_ground_truth} imágenes (esperado)")
    print("Dominios:")
    for domain, count in domains.items():
        print(f"- {domain if pd.notna(domain) else 'No disponible'}: {count}")
    print(f"Formatos de imagen: {', '.join(map(str, sorted(index_df['image_format'].dropna().unique()))) or 'No disponible'}")
    print(f"Formatos de máscara: {', '.join(map(str, sorted(index_df['mask_format'].dropna().unique()))) or 'No disponible'}")
    print(f"Resoluciones: {', '.join(sorted(resolutions.unique())) if not resolutions.empty else 'No disponible'}")
    print(f"Resolución más frecuente: {resolutions.mode().iloc[0] if not resolutions.empty else 'No disponible'}")
    channels = ", ".join(str(int(value)) for value in sorted(index_df["channels"].dropna().unique()))
    print(f"Canales: {channels or 'No disponible'}")
    print(f"Valores únicos observados en máscaras: {observed or 'No disponible'}")
    print(f"Ignore index: {IGNORE_INDEX}")
    print(f"Píxeles ignore: {ignore_pixels}")
    print(f"Porcentaje ignore (todos los píxeles): {100 * ignore_pixels / total_pixels:.6f}%" if total_pixels else "Porcentaje ignore: No disponible")
    observed_valid = sorted(set(observed) & set(CLASS_NAMES))
    print(f"Número de clases válidas observadas: {len(observed_valid)}")
    print(f"Clases observadas: {', '.join(CLASS_NAMES[i] for i in observed_valid) or 'No disponible'}")
    print("Distribución de píxeles por clase (denominador: píxeles válidos 1-7):")
    for _, row in class_df.iterrows():
        print(f"- {row['class_name']}: {int(row['pixel_count'])} ({row['percentage_valid_pixels']:.6f}%)")
    print("Presencia de clases por imagen (denominador: máscaras legibles):")
    for _, row in class_df.iterrows():
        print(f"- {row['class_name']}: {int(row['images_present'])} ({row['percentage_images_present']:.6f}%)")
    if not masks.empty:
        values = masks["num_valid_classes_present"]
        print(f"Número promedio de clases presentes por imagen: {values.mean():.6f}")
        print(f"mínimo: {int(values.min())}")
        print(f"mediana: {values.median():.6f}")
        print(f"máximo: {int(values.max())}")
    else:
        print("Número promedio de clases presentes por imagen: No disponible")
    print(f"Problemas reales encontrados: {len(issues_df)}")
    for issue_type in ISSUE_TYPES:
        print(f"- {issue_type}: {int(issue_counts.get(issue_type, 0))}")
    print(f"Duplicados exactos (grupos SHA-256): {exact_groups}")
    print(f"Duplicados entre splits (grupos SHA-256): {cross_groups}")


def run_full_analysis(
    dataset_root: Path = DATASET_ROOT,
    output_dir: Path = OUTPUT_DIR,
    figures_dir: Path | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Ejecuta toda la Entrega 1 y retorna sus DataFrames para el notebook."""
    root = find_dataset_root(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = figures_dir or output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("Estructura real resumida:\n")
    print(summarized_tree(root))

    index_df, issues_df, auxiliary_df = build_dataset_index(root)
    if index_df.empty:
        raise RuntimeError(f"No se descubrieron imágenes bajo {root}")
    duplicates_df = detect_duplicates(index_df)
    index_df = add_duplicate_flags(index_df, duplicates_df)
    issues_df = add_duplicate_issues(issues_df, duplicates_df)
    class_df, presence_df = make_class_tables(index_df)
    domain_df, split_df, domain_distribution_df, split_distribution_df = make_group_tables(index_df)
    summary_df = make_summary_table(index_df, issues_df, duplicates_df)
    validation_df = validate_analysis(index_df, class_df, domain_distribution_df, split_distribution_df)

    tables = {
        "dataset_index.csv": index_df,
        "dataset_issues.csv": issues_df,
        "dataset_summary.csv": summary_df,
        "class_pixel_counts.csv": class_df,
        "class_image_presence.csv": presence_df,
        "split_counts.csv": split_df,
        "domain_counts.csv": domain_df,
        "duplicates.csv": duplicates_df,
        "class_distribution_by_domain.csv": domain_distribution_df,
        "class_distribution_by_split.csv": split_distribution_df,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8")

    figure_status_df = generate_figures(
        index_df,
        class_df,
        presence_df,
        split_df,
        domain_df,
        domain_distribution_df,
        root,
        figures_dir,
        random_seed,
    )
    return {
        "dataset_root": root,
        "dataset_index_df": index_df,
        "dataset_issues_df": issues_df,
        "auxiliary_files_df": auxiliary_df,
        "duplicates_df": duplicates_df,
        "class_pixel_counts_df": class_df,
        "class_image_presence_df": presence_df,
        "domain_counts_df": domain_df,
        "split_counts_df": split_df,
        "class_distribution_by_domain_df": domain_distribution_df,
        "class_distribution_by_split_df": split_distribution_df,
        "dataset_summary_df": summary_df,
        "figure_status_df": figure_status_df,
        "validation_df": validation_df,
        "experimental_design": experimental_design_text(index_df, duplicates_df),
    }
