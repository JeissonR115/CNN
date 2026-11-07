from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import src.train as train_module


class TrainingPipelineTests(unittest.TestCase):
    @staticmethod
    def create_dataset(root: Path) -> None:
        for class_index, class_name in enumerate(("buildings", "forest", "sea")):
            class_dir = root / "seg_train" / class_name
            class_dir.mkdir(parents=True)
            for image_index in range(10):
                pixels = np.full(
                    (24, 32, 3),
                    (class_index * 70, image_index * 10, 40),
                    dtype=np.uint8,
                )
                Image.fromarray(pixels).save(class_dir / f"{image_index}.png")

    def test_train_returns_model_history_and_ordered_classes(self) -> None:
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            self.create_dataset(dataset_root)
            model_path = root / "models" / "cnn_classifier.keras"
            names_path = root / "models" / "class_names.json"
            with (
                patch.object(train_module, "MODEL_PATH", model_path),
                patch.object(train_module, "CLASS_NAMES_PATH", names_path),
            ):
                model, history, class_names = train_module.train(
                    dataset_root, epochs=1
                )

            self.assertEqual(class_names, ("buildings", "forest", "sea"))
            self.assertEqual(model.output_shape, (None, 3))
            self.assertEqual(len(history.epoch), 1)
            self.assertTrue(model_path.is_file())
            self.assertTrue(names_path.is_file())
            reloaded = tf.keras.models.load_model(model_path)
            self.assertEqual(reloaded.output_shape, (None, 3))

    def test_cli_reports_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    train_module.main(["--data-dir", str(missing)])

            self.assertEqual(error.exception.code, 2)
            self.assertIn("does not exist", stderr.getvalue())

    def test_module_cli_runs_from_repository_root_and_reports_invalid_data(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--data-dir",
                    str(missing),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
