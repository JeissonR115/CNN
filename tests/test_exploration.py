import pickle
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from src.exploration import (
    CIFAR10_CLASSES,
    explore_dataset,
    plot_examples,
    read_record_image,
    select_examples,
)


class ExploreDatasetTests(unittest.TestCase):
    def create_image(self, path: Path, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(20, 40, 60)).save(path)

    def test_reports_counts_dimensions_and_unreadable_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("buildings", "forest", "sea"):
                self.create_image(
                    root / "seg_train" / class_name / "one.png", (12, 8)
                )
                self.create_image(
                    root / "seg_train" / class_name / "two.png", (16, 8)
                )
            (root / "seg_train" / "sea" / "broken.jpg").write_bytes(b"not an image")
            self.create_image(root / "seg_pred" / "one.png", (10, 10))

            report = explore_dataset(root)

            self.assertTrue(report.class_count_valid)
            self.assertEqual(report.class_names, ("buildings", "forest", "sea"))
            self.assertEqual(report.partitions["seg_train"]["sea"], 2)
            self.assertEqual(sum(report.partitions["seg_train"].values()), 6)
            self.assertEqual(report.dimensions, {(12, 8): 3, (16, 8): 3})
            self.assertEqual(report.unlabeled_counts, {"seg_pred": 1})
            self.assertEqual(len(report.unreadable_files), 1)
            self.assertIn("Fuente: Carpetas de imágenes etiquetadas", report.summary())

    def test_supports_nested_partition_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("buildings", "forest", "sea"):
                self.create_image(
                    root / "seg_train" / "seg_train" / class_name / "one.png",
                    (8, 8),
                )

            report = explore_dataset(root)

            self.assertEqual(set(report.partitions), {"seg_train"})
            self.assertEqual(report.total_labeled_images, 3)
            self.assertEqual(report.dimensions, {(8, 8): 3})

    def test_candidate_class_names_fit_the_supported_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            class_names = (
                "buildings",
                "forest",
                "glacier",
                "mountain",
                "sea",
                "street",
            )
            for class_name in class_names:
                self.create_image(
                    root / "seg_train" / class_name / "one.png", (8, 8)
                )

            report = explore_dataset(root)

            self.assertTrue(report.class_count_valid)
            self.assertEqual(report.class_names, class_names)

    def test_reads_cifar10_batches_and_plots_labeled_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "batches.meta").write_bytes(
                pickle.dumps(
                    {
                        b"label_names": [
                            class_name.encode("utf-8") for class_name in CIFAR10_CLASSES
                        ]
                    }
                )
            )
            labels = list(range(10)) * 2
            batch_data = np.arange(20 * 3072, dtype=np.uint8).reshape(20, 3072)
            for batch_name in (
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
                "test_batch",
            ):
                (root / batch_name).write_bytes(
                    pickle.dumps({b"data": batch_data, b"labels": labels})
                )

            report = explore_dataset(root)

            self.assertTrue(report.class_count_valid)
            self.assertEqual(report.class_names, CIFAR10_CLASSES)
            self.assertEqual(report.total_labeled_images, 120)
            self.assertEqual(report.partitions["data_batch_1"]["airplane"], 2)
            self.assertEqual(report.partitions["test_batch"]["truck"], 2)
            self.assertEqual(report.dimensions, {(32, 32): 120})
            self.assertIn("Fuente: CIFAR-10:", report.summary())
            selected = select_examples(report)
            self.assertTrue(all(selected.values()))
            example = read_record_image(selected["airplane"])
            self.assertEqual(example.size, (32, 32))
            self.assertEqual(example.mode, "RGB")
            figure = plot_examples(report)
            self.assertEqual(len(figure.axes), 10)
            matplotlib.pyplot.close(figure)

    def test_rejects_class_counts_outside_the_supported_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("buildings", "forest"):
                self.create_image(root / "train" / class_name / "one.png", (8, 8))

            report = explore_dataset(root)

            self.assertFalse(report.class_count_valid)
            self.assertIn("Bloqueo", report.summary())

    def test_missing_dataset_is_an_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"

            with self.assertRaises(FileNotFoundError):
                explore_dataset(missing)

    def test_example_selection_is_reproducible_and_plot_handles_empty_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("buildings", "forest", "sea"):
                self.create_image(root / "train" / class_name / "one.png", (8, 8))
                self.create_image(root / "train" / class_name / "two.png", (8, 8))
            (root / "train" / "forest" / "broken.png").write_bytes(b"invalid")
            (root / "train" / "glacier").mkdir()

            report = explore_dataset(root)

            self.assertEqual(select_examples(report), select_examples(report))
            self.assertIsNone(select_examples(report)["glacier"])
            figure = plot_examples(report)
            self.assertEqual(len(figure.axes), 4)
            matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
