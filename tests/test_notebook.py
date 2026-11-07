from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        self.assertLess(
            source.index("train(DATASET_DIR"),
            source.index("test_probabilities = model.predict(test_data"),
        )
        self.assertIn("from src.predict import predict", source)
        self.assertIn("replace=False", source)
        self.assertIn("https://github.com/JeissonR115/CNN.git", source)
        self.assertIn("pip", source)
        self.assertIn("cifar-10-python.tar.gz", source)
        self.assertIn("files.download", source)

    def test_notebook_prediction_sampling_is_distinct_reproducible_and_capped(self) -> None:
        import matplotlib.pyplot as plt

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "sample_count = min(10" in "".join(cell["source"])
        )
        class_names = ("buildings", "forest", "sea")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for index in range(15):
                image_path = root / f"{index}.png"
                Image.new("RGB", (8, 8), color=(index, 20, 40)).save(image_path)
                records.append(
                    SimpleNamespace(
                        path=image_path,
                        class_name=class_names[index % len(class_names)],
                    )
                )
            namespace = {
                "DATASET_AVAILABLE": True,
                "splits": SimpleNamespace(test=tuple(records)),
                "class_names": class_names,
                "test_probabilities": np.full((15, 3), 1 / 3, dtype=np.float32),
                "np": np,
                "plt": plt,
                "Image": Image,
                "display": lambda figure: None,
            }
            exec(cell_source, namespace)
            first_indices = namespace["selected_indices"].copy()
            exec(cell_source, namespace)
            second_indices = namespace["selected_indices"].copy()
            self.assertEqual(len(first_indices), 10)
            self.assertEqual(len(set(first_indices.tolist())), 10)
            np.testing.assert_array_equal(first_indices, second_indices)
            plt.close("all")

            namespace["splits"] = SimpleNamespace(test=tuple(records[:4]))
            namespace["test_probabilities"] = np.full((4, 3), 1 / 3, dtype=np.float32)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source, namespace)
            self.assertEqual(len(namespace["selected_indices"]), 4)
            self.assertIn("no se alcanzó el mínimo de 10", output.getvalue())
            plt.close("all")

    def test_notebook_reports_controlled_test_metrics_and_handles_empty_test(self) -> None:
        import matplotlib.pyplot as plt

        from src.exploration import ImageRecord

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "test_probabilities = model.predict(test_data" in "".join(cell["source"])
        )
        class_names = ("buildings", "forest", "sea")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for index, class_name in enumerate(class_names):
                image_path = root / f"{index}.png"
                Image.new("RGB", (8, 8), color=(index * 50, 20, 40)).save(image_path)
                records.append(
                    ImageRecord(image_path, "seg_train", class_name, 8, 8)
                )

            class ControlledModel:
                def predict(self, dataset, verbose=0):
                    return np.asarray(
                        [
                            [0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.7, 0.2],
                        ],
                        dtype=np.float32,
                    )

            namespace = {
                "DATASET_AVAILABLE": True,
                "splits": SimpleNamespace(test=tuple(records)),
                "class_names": class_names,
                "model": ControlledModel(),
                "np": np,
                "plt": plt,
                "display": lambda figure: None,
            }
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source, namespace)
            np.testing.assert_array_equal(
                namespace["matrix"],
                np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0]]),
            )
            self.assertIn("Test accuracy: 0.6667", output.getvalue())
            self.assertIn("support", output.getvalue())
            plt.close("all")

            namespace["splits"] = SimpleNamespace(test=())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source, namespace)
            self.assertIn("no hay ejemplos etiquetados en test", output.getvalue())
            self.assertEqual(namespace["test_probabilities"].shape, (0, 3))

    def test_notebook_validates_and_plots_training_history_series(self) -> None:
        import matplotlib.pyplot as plt

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "required_history = " in "".join(cell["source"])
        )
        history = SimpleNamespace(
            history={
                "loss": [0.8, 0.5],
                "val_loss": [0.9, 0.6],
                "accuracy": [0.6, 0.8],
                "val_accuracy": [0.5, 0.7],
            }
        )
        namespace = {
            "DATASET_AVAILABLE": True,
            "history": history,
            "np": np,
            "plt": plt,
            "display": lambda figure: None,
        }
        exec(cell_source, namespace)
        self.assertEqual(len(namespace["axes"][0].lines), 2)
        np.testing.assert_array_equal(
            namespace["axes"][1].lines[0].get_ydata(), [0.6, 0.8]
        )
        plt.close("all")

        namespace["history"] = SimpleNamespace(history={"loss": [1.0]})
        with self.assertRaisesRegex(ValueError, "no contiene las series"):
            exec(cell_source, namespace)
        plt.close("all")

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
