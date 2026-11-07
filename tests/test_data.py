from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import (
    augmentation_preview,
    build_augmentation,
    load_labeled_records,
    make_dataset,
    plot_augmentation_examples,
    preprocess_image,
    split_dataset,
)
from src.exploration import ImageRecord


class ImagePreprocessingTests(unittest.TestCase):
    def test_preprocessing_converts_modes_resizes_and_normalizes(self) -> None:
        for mode, color in (("RGB", (0, 127, 255)), ("L", 128), ("RGBA", (1, 2, 3, 4))):
            with self.subTest(mode=mode):
                result = preprocess_image(Image.new(mode, (19, 7), color=color))
                self.assertEqual(result.shape, (128, 128, 3))
                self.assertEqual(result.dtype, np.float32)
                self.assertGreaterEqual(float(result.min()), 0.0)
                self.assertLessEqual(float(result.max()), 1.0)

        extremes = preprocess_image(Image.new("RGB", (2, 2), color=(0, 127, 255)))
        np.testing.assert_allclose(extremes[0, 0], [0.0, 127 / 255, 1.0])

    def test_preprocessing_rejects_non_pil_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "PIL.Image.Image"):
            preprocess_image(object())


class DatasetSplitTests(unittest.TestCase):
    @staticmethod
    def records(count_per_class: int = 20) -> tuple[ImageRecord, ...]:
        return tuple(
            ImageRecord(
                path=Path(f"/dataset/{class_name}/{index}.jpg"),
                partition="train",
                class_name=class_name,
                width=150,
                height=150,
            )
            for class_name in ("buildings", "forest", "sea")
            for index in range(count_per_class)
        )

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        records = self.records()
        first = split_dataset(records)
        second = split_dataset(records)

        self.assertEqual(first, second)
        self.assertEqual(first.class_names, ("buildings", "forest", "sea"))
        self.assertEqual(
            (len(first.train), len(first.validation), len(first.test)),
            (42, 9, 9),
        )
        partitions = [
            {record.path for record in subset}
            for subset in (first.train, first.validation, first.test)
        ]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        for subset in (first.train, first.validation, first.test):
            self.assertEqual(
                {record.class_name for record in subset}, set(first.class_names)
            )

    def test_split_reports_insufficient_class_counts(self) -> None:
        records = tuple(
            record
            for record in self.records(count_per_class=8)
            if record.class_name != "sea" or record.path.name == "0.jpg"
        )
        with self.assertRaisesRegex(ValueError, "stratified 70/15/15"):
            split_dataset(records)

    def test_dataset_discovery_excludes_unlabeled_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("buildings", "forest", "sea"):
                directory = root / "seg_train" / class_name
                directory.mkdir(parents=True)
                for index in range(8):
                    Image.new("RGB", (8, 8)).save(directory / f"{index}.png")
            prediction_dir = root / "seg_pred"
            prediction_dir.mkdir()
            Image.new("RGB", (8, 8)).save(prediction_dir / "unlabeled.png")

            records = load_labeled_records(root)

            self.assertEqual(len(records), 24)
            self.assertTrue(all(record.partition == "seg_train" for record in records))


class AugmentationTests(unittest.TestCase):
    def test_training_augmentation_is_finite_and_clipped(self) -> None:
        import tensorflow as tf

        image = np.linspace(0.0, 1.0, 128 * 128 * 3, dtype=np.float32).reshape(
            (1, 128, 128, 3)
        )
        augmenter = build_augmentation(seed=7)
        augmented = tf.clip_by_value(augmenter(image, training=True), 0.0, 1.0)

        self.assertEqual(tuple(augmented.shape), (1, 128, 128, 3))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(augmented))))
        self.assertGreaterEqual(float(tf.reduce_min(augmented)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(augmented)), 1.0)

    def test_augmentation_preview_is_reproducible_and_shows_all_operations(self) -> None:
        image = Image.new("RGB", (37, 19), color=(40, 100, 220))
        first = augmentation_preview(image, seed=17)
        second = augmentation_preview(image, seed=17)

        self.assertEqual(
            tuple(first),
            (
                "Original",
                "Flip horizontal",
                "Rotación ±15°",
                "Zoom ±10 %",
                "Contraste 0.8-1.2",
                "Combinación",
            ),
        )
        for name in first:
            with self.subTest(transformation=name):
                self.assertEqual(first[name].shape, (128, 128, 3))
                self.assertTrue(np.isfinite(first[name]).all())
                self.assertGreaterEqual(float(first[name].min()), 0.0)
                self.assertLessEqual(float(first[name].max()), 1.0)
                np.testing.assert_array_equal(first[name], second[name])

    def test_augmentation_grid_has_one_reproducible_example_per_class(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            class_names = ("buildings", "forest", "sea")
            for class_index, class_name in enumerate(class_names):
                for image_index in range(2):
                    path = root / class_name / f"{image_index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB",
                        (20, 10),
                        color=(class_index * 70, image_index * 100, 50),
                    ).save(path)
                    records.append(
                        ImageRecord(path, "train", class_name, 20, 10)
                    )

            first = plot_augmentation_examples(records, class_names, seed=42)
            second = plot_augmentation_examples(records, class_names, seed=42)
            self.assertEqual(len(first.axes), 18)
            for first_axis, second_axis in zip(first.axes, second.axes):
                np.testing.assert_array_equal(
                    first_axis.images[0].get_array(),
                    second_axis.images[0].get_array(),
                )
            plt.close(first)
            plt.close(second)

    def test_only_augmented_dataset_has_random_transformations(self) -> None:
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index in range(2):
                path = root / f"{index}.png"
                Image.fromarray(
                    np.tile(
                        np.arange(128, dtype=np.uint8)[None, :, None], (128, 1, 3)
                    )
                ).save(path)
                paths.append(
                    ImageRecord(path, "train", "buildings", 128, 128)
                )
            class_names = ("buildings",)

            base = make_dataset(paths, class_names, batch_size=2, augment=False)
            training = make_dataset(
                paths, class_names, batch_size=2, augment=True, seed=11
            )
            base_images = next(iter(base))[0]
            training_images = next(iter(training))[0]

            self.assertTrue(
                bool(
                    tf.reduce_all(
                        tf.math.equal(base_images[0], base_images[1])
                    )
                )
            )
            self.assertFalse(
                bool(
                    tf.reduce_all(
                        tf.math.equal(training_images[0], training_images[1])
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
