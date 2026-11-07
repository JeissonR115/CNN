from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.exploration import ImageRecord, explore_dataset, read_record_image

IMAGE_SIZE = (128, 128)
SPLIT_SEED = 42
MIN_CLASSES = 3
MAX_CLASSES = 10


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]
    class_names: tuple[str, ...]


def preprocess_image(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")
    try:
        image.load()
        rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (OSError, ValueError) as error:
        raise ValueError("Image could not be decoded and preprocessed.") from error
    result = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    if result.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(f"Preprocessed image has an invalid shape: {result.shape}.")
    return result


def load_labeled_records(dataset_root: str | Path) -> tuple[ImageRecord, ...]:
    report = explore_dataset(dataset_root)
    if not report.records:
        raise ValueError(f"No readable labeled images were found in {report.root}.")
    if not report.class_count_valid:
        raise ValueError(
            "The dataset must contain between "
            f"{MIN_CLASSES} and {MAX_CLASSES} readable labeled classes; "
            f"found {len(report.class_names)}."
        )
    return tuple(
        sorted(
            report.records,
            key=lambda record: (str(record.path), record.sample_index or 0),
        )
    )


def split_dataset(records: Sequence[ImageRecord]) -> DatasetSplits:
    if not records:
        raise ValueError("Cannot split an empty labeled dataset.")

    record_list = tuple(records)
    identities = [
        (record.path.resolve(), record.sample_index) for record in record_list
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Labeled dataset contains duplicate image records.")

    class_names = tuple(sorted({record.class_name for record in record_list}))
    if any(not name for name in class_names):
        raise ValueError("Every labeled image must have a non-empty class name.")
    if not MIN_CLASSES <= len(class_names) <= MAX_CLASSES:
        raise ValueError(
            f"Expected between {MIN_CLASSES} and {MAX_CLASSES} classes; "
            f"found {len(class_names)}."
        )

    cifar_records = [record for record in record_list if record.sample_index is not None]
    if cifar_records and len(cifar_records) != len(record_list):
        raise ValueError("CIFAR-10 records cannot be mixed with image-folder records.")
    if cifar_records:
        official_test = [
            index
            for index, record in enumerate(record_list)
            if record.partition == "test_batch"
        ]
        train_pool = [
            index
            for index, record in enumerate(record_list)
            if record.partition != "test_batch"
        ]
        if not official_test or not train_pool:
            raise ValueError(
                "CIFAR-10 requires training batches and the official test_batch."
            )
        training_labels = np.asarray(
            [record_list[index].class_name for index in train_pool]
        )
        try:
            train_indices, validation_indices = train_test_split(
                train_pool,
                test_size=0.15,
                random_state=SPLIT_SEED,
                stratify=training_labels,
            )
        except ValueError as error:
            raise ValueError(
                "Cannot create a stratified CIFAR-10 train/validation split. "
                "Each class needs enough examples in the training batches."
            ) from error
        return DatasetSplits(
            train=tuple(record_list[index] for index in sorted(train_indices)),
            validation=tuple(
                record_list[index] for index in sorted(validation_indices)
            ),
            test=tuple(record_list[index] for index in official_test),
            class_names=class_names,
        )

    labels = np.asarray([record.class_name for record in record_list])
    indices = np.arange(len(record_list))
    counts = {
        class_name: int(np.count_nonzero(labels == class_name))
        for class_name in class_names
    }
    try:
        train_indices, remainder_indices = train_test_split(
            indices,
            test_size=0.3,
            random_state=SPLIT_SEED,
            stratify=labels,
        )
        validation_indices, test_indices = train_test_split(
            remainder_indices,
            test_size=0.5,
            random_state=SPLIT_SEED,
            stratify=labels[remainder_indices],
        )
    except ValueError as error:
        formatted_counts = ", ".join(
            f"{class_name}={count}" for class_name, count in counts.items()
        )
        raise ValueError(
            "Cannot create a stratified 70/15/15 split while preserving every "
            "class in all subsets. Add labeled images to the underrepresented "
            f"classes. Current counts: {formatted_counts}."
        ) from error

    return DatasetSplits(
        train=tuple(record_list[index] for index in sorted(train_indices)),
        validation=tuple(
            record_list[index] for index in sorted(validation_indices)
        ),
        test=tuple(record_list[index] for index in sorted(test_indices)),
        class_names=class_names,
    )


def build_augmentation(seed: int = SPLIT_SEED):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(
                factor=15 / 360,
                fill_mode="reflect",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.1, 0.1),
                width_factor=(-0.1, 0.1),
                fill_mode="reflect",
                seed=seed + 2,
            ),
            tf.keras.layers.RandomContrast(factor=0.2, seed=seed + 3),
        ],
        name="training_augmentation",
    )


def augmentation_preview(
    image: Image.Image, *, seed: int = SPLIT_SEED
) -> dict[str, np.ndarray]:
    import tensorflow as tf

    original = preprocess_image(image)
    batch = tf.convert_to_tensor(original[None, ...])
    transformations = {
        "Original": original,
        "Flip horizontal": tf.image.flip_left_right(batch)[0],
        "Rotación ±15°": tf.keras.layers.RandomRotation(
            15 / 360, fill_mode="reflect", seed=seed + 1
        )(batch, training=True)[0],
        "Zoom ±10 %": tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=seed + 2,
        )(batch, training=True)[0],
        "Contraste 0.8-1.2": tf.keras.layers.RandomContrast(
            0.2, seed=seed + 3
        )(batch, training=True)[0],
        "Combinación": build_augmentation(seed)(batch, training=True)[0],
    }
    return {
        name: np.clip(
            value.numpy() if hasattr(value, "numpy") else value, 0.0, 1.0
        ).astype(np.float32)
        for name, value in transformations.items()
    }


def plot_augmentation_examples(
    records: Sequence[ImageRecord],
    class_names: Sequence[str],
    *,
    seed: int = SPLIT_SEED,
):
    import matplotlib.pyplot as plt

    examples_by_class: dict[str, list[ImageRecord]] = {
        class_name: [] for class_name in class_names
    }
    for record in records:
        if record.class_name in examples_by_class:
            examples_by_class[record.class_name].append(record)
    generator = random.Random(seed)
    selected = []
    for class_name in class_names:
        candidates = sorted(
            examples_by_class[class_name], key=lambda record: str(record.path)
        )
        if not candidates:
            raise ValueError(
                f"Training partition has no readable example for {class_name!r}."
            )
        selected.append(generator.choice(candidates))

    columns = tuple(
        (
            "Original",
            "Flip horizontal",
            "Rotación ±15°",
            "Zoom ±10 %",
            "Contraste 0.8-1.2",
            "Combinación",
        )
    )
    figure, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(3 * len(columns), 3 * len(selected)),
        squeeze=False,
    )
    for row, record in enumerate(selected):
        examples = augmentation_preview(
            read_record_image(record), seed=seed + row
        )
        for column, title in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(examples[title])
            axis.set_title(title)
            axis.axis("off")
        axes[row, 0].set_ylabel(record.class_name)
    figure.tight_layout()
    return figure


def make_dataset(
    records: Sequence[ImageRecord],
    class_names: Sequence[str],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = SPLIT_SEED,
):
    import tensorflow as tf

    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    if not records:
        raise ValueError("Cannot create a TensorFlow dataset from no images.")
    names = tuple(class_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("class_names must contain unique, non-empty names.")
    class_indices = {name: index for index, name in enumerate(names)}
    for record in records:
        if record.class_name not in class_indices:
            raise ValueError(
                f"Image class {record.class_name!r} is missing from class_names."
            )

    record_list = tuple(records)

    def samples():
        for record in record_list:
            try:
                pixels = preprocess_image(read_record_image(record))
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"Could not read labeled image {record.path} "
                    f"(sample {record.sample_index})."
                ) from error
            yield pixels, np.int32(class_indices[record.class_name])

    dataset = tf.data.Dataset.from_generator(
        samples,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(record_list), 1024),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    if augment:
        augmenter = build_augmentation(seed)
        dataset = dataset.map(
            lambda image, label: (
                tf.clip_by_value(augmenter(image, training=True), 0.0, 1.0),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    dataset = dataset.batch(batch_size).apply(
        tf.data.experimental.assert_cardinality(
            (len(record_list) + batch_size - 1) // batch_size
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
