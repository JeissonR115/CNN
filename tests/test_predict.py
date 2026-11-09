from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.data import preprocess_image
from src.predict import _load_model_and_classes, predict, predict_all


class PredictionTests(unittest.TestCase):
    def tearDown(self) -> None:
        _load_model_and_classes.cache_clear()

    def test_prediction_preprocesses_and_maps_the_highest_probability(self) -> None:
        class FakeModel:
            output_shape = (None, 3)

            def __call__(self, values, training: bool = False):
                self.values = values
                self.training = training
                return np.asarray([[0.1, 0.75, 0.15]], dtype=np.float32)

        model = FakeModel()
        image = Image.new("L", (21, 9), color=128)
        with patch(
            "src.predict._load_model_and_classes",
            return_value=(model, ("buildings", "forest", "sea")),
        ):
            class_name, confidence = predict(image)

        self.assertEqual(class_name, "forest")
        self.assertEqual(confidence, 0.75)
        self.assertTrue(model.training is False)
        np.testing.assert_array_equal(
            model.values[0], preprocess_image(image)
        )
        self.assertEqual(model.values.shape, (1, 32, 32, 3))

    def test_predict_all_returns_all_classes_in_descending_probability_order(self) -> None:
        class FakeModel:
            output_shape = (None, 3)

            def __call__(self, values, training: bool = False):
                return np.asarray([[0.1, 0.75, 0.15]], dtype=np.float32)

        with patch(
            "src.predict._load_model_and_classes",
            return_value=(FakeModel(), ("airplane", "bird", "cat")),
        ):
            predictions = predict_all(Image.new("RGB", (32, 32)))

        self.assertEqual(
            tuple(name for name, _ in predictions),
            ("bird", "cat", "airplane"),
        )
        np.testing.assert_allclose(
            [probability for _, probability in predictions],
            [0.75, 0.15, 0.1],
        )

    def test_non_pil_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "PIL.Image.Image"):
            predict(object())

    def test_missing_artifacts_and_malformed_class_names_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(FileNotFoundError, "model artifact is missing"):
                _load_model_and_classes(
                    str(root / "missing.keras"), str(root / "classes.json")
                )

            model_path = root / "model.keras"
            class_names_path = root / "classes.json"
            model_path.touch()
            class_names_path.write_text("{", encoding="utf-8")
            with patch("tensorflow.keras.models.load_model", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "as JSON"):
                    _load_model_and_classes(
                        str(model_path), str(class_names_path)
                    )

    def test_saved_artifacts_are_loaded_once_and_output_dimension_is_checked(self) -> None:
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.keras"
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(32, 32, 3)),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )
            model.save(model_path)
            names_path = root / "classes.json"
            names_path.write_text(
                json.dumps(["buildings", "forest", "sea"]), encoding="utf-8"
            )

            loaded, names = _load_model_and_classes(str(model_path), str(names_path))
            loaded_again, names_again = _load_model_and_classes(
                str(model_path), str(names_path)
            )
            self.assertIs(loaded, loaded_again)
            self.assertEqual(names, names_again)

            names_path.write_text(json.dumps(["buildings", "forest"]), encoding="utf-8")
            _load_model_and_classes.cache_clear()
            with self.assertRaisesRegex(ValueError, "output count"):
                _load_model_and_classes(str(model_path), str(names_path))


if __name__ == "__main__":
    unittest.main()
