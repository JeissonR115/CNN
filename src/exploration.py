from __future__ import annotations

import pickle
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
UNLABELED_PARTITIONS = {"pred", "prediction", "predictions", "seg_pred"}
PARTITION_DIRECTORIES = {
    "seg_test",
    "seg_train",
    "test",
    "train",
    "val",
    "valid",
    "validation",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    partition: str
    class_name: str
    width: int
    height: int
    sample_index: int | None = None


@dataclass(frozen=True)
class DatasetReport:
    root: Path
    partitions: dict[str, dict[str, int]]
    unlabeled_counts: dict[str, int]
    dimensions: dict[tuple[int, int], int]
    unreadable_files: tuple[Path, ...]
    records: tuple[ImageRecord, ...]
    class_names: tuple[str, ...]
    min_classes: int
    max_classes: int

    @property
    def class_count_valid(self) -> bool:
        return self.min_classes <= len(self.class_names) <= self.max_classes

    @property
    def total_labeled_images(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        source = (
            f"CIFAR-10: {DATASET_URL}"
            if any(record.sample_index is not None for record in self.records)
            else "Carpetas de imágenes etiquetadas"
        )
        lines = [
            f"Fuente: {source}",
            f"Raíz medida: {self.root}",
            f"Clases con imágenes legibles: {len(self.class_names)}",
            (
                f"Validación del rango {self.min_classes}-{self.max_classes}: "
                f"{'válida' if self.class_count_valid else 'no válida'}"
            ),
            f"Imágenes etiquetadas legibles: {self.total_labeled_images}",
            "",
            "Imágenes por partición y clase (solo archivos decodificables):",
        ]
        if self.partitions:
            for partition, counts in sorted(self.partitions.items()):
                lines.append(
                    f"- {partition}: "
                    + ", ".join(
                        f"{name}={count}" for name, count in sorted(counts.items())
                    )
                    + f" (total={sum(counts.values())})"
                )
        else:
            lines.append("- No se encontraron directorios de clases.")
        lines.append("")
        lines.append("Dimensiones observadas (ancho × alto):")
        if self.dimensions:
            lines.extend(
                f"- {width} × {height}: {count}"
                for (width, height), count in sorted(self.dimensions.items())
            )
        else:
            lines.append("- No hay imágenes etiquetadas legibles para medir.")
        lines.append(f"Archivos de imagen ilegibles: {len(self.unreadable_files)}")
        if self.unlabeled_counts:
            lines.append(
                "Archivos en particiones sin etiquetas (excluidos de los conteos "
                "por clase): "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(self.unlabeled_counts.items())
                )
            )
        if not self.class_count_valid:
            lines.append(
                "Bloqueo: el dataset debe tener entre "
                f"{self.min_classes} y {self.max_classes} clases no vacías."
            )
        return "\n".join(lines)


def _image_files(directory: Path, *, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _class_directories(directory: Path) -> list[Path]:
    class_directories: list[Path] = []
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if _image_files(child, recursive=False) or not any(child.iterdir()):
            class_directories.append(child)
    return sorted(class_directories)


def _partition_name(root: Path, class_root: Path) -> str:
    relative = class_root.relative_to(root)
    return relative.parts[0] if relative.parts else "dataset"


@lru_cache(maxsize=6)
def _read_cifar_batch(batch_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    try:
        with batch_path.open("rb") as batch_file:
            batch = pickle.load(batch_file, encoding="bytes")
    except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
        raise ValueError(f"Could not read CIFAR-10 batch {batch_path}.") from error

    data = np.asarray(batch.get(b"data"))
    labels = batch.get(b"labels")
    if (
        data.ndim != 2
        or data.shape[1] != 3 * 32 * 32
        or labels is None
        or len(labels) != len(data)
        or any(not isinstance(label, (int, np.integer)) or not 0 <= label < 10 for label in labels)
    ):
        raise ValueError(f"CIFAR-10 batch has an invalid data or label format: {batch_path}.")
    return data, tuple(int(label) for label in labels)


def read_record_image(record: ImageRecord) -> Image.Image:
    sample_index = getattr(record, "sample_index", None)
    if sample_index is None:
        try:
            with Image.open(record.path) as image:
                return image.convert("RGB").copy()
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise ValueError(f"Could not read image {record.path}.") from error

    data, labels = _read_cifar_batch(record.path)
    if not 0 <= sample_index < len(data):
        raise ValueError(
            f"CIFAR-10 sample index is out of range for {record.path}: "
            f"{sample_index}."
        )
    pixels = data[sample_index].reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(pixels.astype(np.uint8, copy=False), mode="RGB")


def _explore_cifar10(
    root: Path, *, min_classes: int, max_classes: int
) -> DatasetReport:
    required = ("batches.meta", *CIFAR10_BATCHES, "test_batch")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(
            f"Incomplete CIFAR-10 dataset at {root}; missing: {', '.join(missing)}."
        )
    try:
        with (root / "batches.meta").open("rb") as metadata_file:
            metadata = pickle.load(metadata_file, encoding="bytes")
    except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
        raise ValueError(f"Could not read CIFAR-10 metadata in {root}.") from error
    label_names = metadata.get(b"label_names")
    if label_names is None:
        raise ValueError(f"CIFAR-10 metadata has no label_names in {root}.")
    class_names = tuple(
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in label_names
    )
    if class_names != CIFAR10_CLASSES:
        raise ValueError("CIFAR-10 metadata has unexpected class names.")

    partitions: dict[str, dict[str, int]] = {}
    records: list[ImageRecord] = []
    for batch_name in (*CIFAR10_BATCHES, "test_batch"):
        batch_path = root / batch_name
        data, labels = _read_cifar_batch(batch_path)
        counts = Counter(class_names[label] for label in labels)
        partitions[batch_name] = {
            class_name: counts.get(class_name, 0) for class_name in class_names
        }
        records.extend(
            ImageRecord(
                path=batch_path,
                partition=batch_name,
                class_name=class_names[label],
                width=32,
                height=32,
                sample_index=index,
            )
            for index, label in enumerate(labels)
        )

    return DatasetReport(
        root=root,
        partitions=partitions,
        unlabeled_counts={},
        dimensions={(32, 32): len(records)},
        unreadable_files=(),
        records=tuple(records),
        class_names=class_names,
        min_classes=min_classes,
        max_classes=max_classes,
    )


def explore_dataset(
    dataset_root: str | Path,
    *,
    min_classes: int = 3,
    max_classes: int = 10,
) -> DatasetReport:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if min_classes < 1 or max_classes < min_classes:
        raise ValueError("Class-count limits must satisfy 1 <= min_classes <= max_classes.")
    if (root / "batches.meta").exists():
        return _explore_cifar10(
            root, min_classes=min_classes, max_classes=max_classes
        )

    search_directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    class_roots: dict[str, dict[str, Path]] = {}
    for candidate in search_directories:
        relative = candidate.relative_to(root)
        if not relative.parts and any(
            child.name.casefold() in PARTITION_DIRECTORIES
            for child in candidate.iterdir()
            if child.is_dir()
        ):
            continue
        if relative.parts and relative.parts[0].casefold() in UNLABELED_PARTITIONS:
            continue
        class_directories = _class_directories(candidate)
        if not class_directories:
            continue
        partition = _partition_name(root, candidate)
        class_roots.setdefault(partition, {}).update(
            {directory.name: directory for directory in class_directories}
        )

    records: list[ImageRecord] = []
    unreadable: list[Path] = []
    partitions: dict[str, dict[str, int]] = {}
    dimensions: Counter[tuple[int, int]] = Counter()
    all_class_names: set[str] = set()

    for partition, directories in sorted(class_roots.items()):
        partitions[partition] = {}
        for class_name, class_directory in sorted(directories.items()):
            all_class_names.add(class_name)
            count = 0
            for image_path in _image_files(class_directory, recursive=True):
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        width, height = image.size
                except (OSError, ValueError, UnidentifiedImageError):
                    unreadable.append(image_path)
                    continue
                records.append(
                    ImageRecord(
                        path=image_path,
                        partition=partition,
                        class_name=class_name,
                        width=width,
                        height=height,
                    )
                )
                dimensions[(width, height)] += 1
                count += 1
            partitions[partition][class_name] = count

    populated_classes = {
        class_name
        for counts in partitions.values()
        for class_name, count in counts.items()
        if count > 0
    }

    unlabeled_counts: dict[str, int] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        if child.name.casefold() in UNLABELED_PARTITIONS:
            count = 0
            for image_path in _image_files(child, recursive=True):
                try:
                    with Image.open(image_path) as image:
                        image.load()
                except (OSError, ValueError, UnidentifiedImageError):
                    unreadable.append(image_path)
                    continue
                count += 1
            unlabeled_counts[child.name] = count

    return DatasetReport(
        root=root,
        partitions=partitions,
        unlabeled_counts=unlabeled_counts,
        dimensions=dict(dimensions),
        unreadable_files=tuple(sorted(unreadable)),
        records=tuple(records),
        class_names=tuple(sorted(populated_classes)),
        min_classes=min_classes,
        max_classes=max_classes,
    )


def select_examples(
    report: DatasetReport, *, seed: int = 42
) -> dict[str, ImageRecord | None]:
    generator = random.Random(seed)
    examples: dict[str, ImageRecord | None] = {}
    for class_name in sorted(
        {record.class_name for record in report.records}
        | {
            class_name
            for counts in report.partitions.values()
            for class_name, count in counts.items()
            if count == 0
        }
    ):
        candidates = sorted(
            (record for record in report.records if record.class_name == class_name),
            key=lambda record: (str(record.path), record.sample_index or 0),
        )
        examples[class_name] = generator.choice(candidates) if candidates else None
    return examples


def plot_examples(report: DatasetReport, *, seed: int = 42):
    examples = select_examples(report, seed=seed)
    if not examples:
        figure, axis = plt.subplots(figsize=(6, 2))
        axis.text(0.5, 0.5, "No labeled classes found", ha="center", va="center")
        axis.axis("off")
        return figure

    figure, axes = plt.subplots(
        1, len(examples), figsize=(3 * len(examples), 3), squeeze=False
    )
    for axis, (class_name, record) in zip(axes[0], examples.items()):
        axis.set_title(class_name)
        axis.axis("off")
        if record is None:
            axis.text(0.5, 0.5, "No readable example", ha="center", va="center")
            continue
        axis.imshow(read_record_image(record))
    figure.tight_layout()
    return figure
