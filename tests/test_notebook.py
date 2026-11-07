from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

import src.predict as predict_module
import src.train as train_module

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "entrenamiento.ipynb"


class TrainingNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_clean_and_contains_executable_cells(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        for index, cell in enumerate(code_cells):
            with self.subTest(cell=index):
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        self.assertLess(source.index("train(DATASET_DIR"), source.index("model.evaluate"))
        self.assertIn("from src.predict import predict", source)
        self.assertIn("replace=False", source)

    def test_notebook_runs_end_to_end_with_a_configured_small_dataset(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            for class_index, class_name in enumerate(("buildings", "forest", "sea")):
                class_dir = dataset_root / "seg_train" / class_name
                class_dir.mkdir(parents=True)
                for image_index in range(10):
                    image = np.full(
                        (20, 24, 3),
                        (class_index * 70, image_index * 10, 40),
                        dtype=np.uint8,
                    )
                    Image.fromarray(image).save(class_dir / f"{image_index}.png")

            model_path = root / "models" / "cnn_classifier.keras"
            names_path = root / "models" / "class_names.json"
            notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
            namespace: dict[str, object] = {}
            with (
                patch.dict(
                    os.environ,
                    {
                        "CNN_DATASET_DIR": str(dataset_root),
                        "CNN_TRAINING_EPOCHS": "1",
                    },
                ),
                patch.object(train_module, "MODEL_PATH", model_path),
                patch.object(train_module, "CLASS_NAMES_PATH", names_path),
                patch.object(predict_module, "MODEL_PATH", model_path),
                patch.object(predict_module, "CLASS_NAMES_PATH", names_path),
                patch("IPython.display.display", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec("".join(cell["source"]), namespace)

            self.assertEqual(namespace["class_names"], ("buildings", "forest", "sea"))
            self.assertEqual(len(namespace["history"].epoch), 1)
            self.assertEqual(namespace["test_probabilities"].shape[1], 3)
            self.assertTrue(model_path.is_file())
            self.assertTrue(names_path.is_file())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
