from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
import src.predict as predict_module
from app import decode_uploaded_image


class ImageUploadTests(unittest.TestCase):
    @staticmethod
    def encoded_image(image_format: str) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (10, 7), color=(30, 60, 90)).save(
            buffer, format=image_format
        )
        return buffer.getvalue()

    @staticmethod
    def clear_prediction_resources() -> None:
        app._prediction_api.clear()
        st.cache_resource.clear()
        predict_module.clear_prediction_cache()

    def test_decodes_supported_jpeg_and_png_as_pil_rgb(self) -> None:
        for image_format in ("JPEG", "PNG"):
            with self.subTest(image_format=image_format):
                image = decode_uploaded_image(self.encoded_image(image_format))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (10, 7))

    def test_rejects_unreadable_or_unsupported_uploads(self) -> None:
        with self.assertRaisesRegex(ValueError, "readable JPEG or PNG"):
            decode_uploaded_image(b"not an image")
        with self.assertRaisesRegex(ValueError, "readable JPEG or PNG"):
            decode_uploaded_image(self.encoded_image("GIF"))

    def test_prediction_resource_is_cached(self) -> None:
        self.clear_prediction_resources()
        with patch("src.predict.predict") as predictor:
            self.assertIs(app._prediction_api(), app._prediction_api())
            self.assertIs(app._prediction_api(), predictor)
        self.clear_prediction_resources()

    def test_streamlit_handles_invalid_uploads_artifact_errors_and_predictions(self) -> None:
        import tensorflow as tf

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "cnn_classifier.keras"
            names_path = root / "class_names.json"
            self.clear_prediction_resources()
            with (
                patch.object(predict_module, "MODEL_PATH", model_path),
                patch.object(predict_module, "CLASS_NAMES_PATH", names_path),
            ):
                at = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "app.py")
                ).run()
                at.file_uploader[0].upload(
                    "broken.png", b"not an image", "image/png"
                ).run()
                self.assertIn(
                    "The uploaded file is not a readable JPEG or PNG image.",
                    [element.value for element in at.error],
                )
                self.assertFalse(at.subheader)

                at.file_uploader[0].upload(
                    "sample.png", self.encoded_image("PNG"), "image/png"
                ).run()
                at.button[0].click().run()
                self.assertTrue(
                    any(
                        "Trained model artifact is missing" in element.value
                        for element in at.error
                    )
                )
                self.assertFalse(at.subheader)

                model = tf.keras.Sequential(
                    [
                        tf.keras.Input(shape=(128, 128, 3)),
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(3, activation="softmax"),
                    ]
                )
                model.save(model_path)
                names_path.write_text(
                    json.dumps(["buildings", "forest", "sea"]), encoding="utf-8"
                )
                predict_module.clear_prediction_cache()
                at.button[0].click().run()

                self.assertFalse(at.error)
                self.assertEqual(len(at.subheader), 1)
                self.assertTrue(
                    at.subheader[0].value.startswith("Predicted class: ")
                )
                self.assertTrue(at.progress)
                self.assertGreaterEqual(at.progress[0].proto.value, 0)
                self.assertLessEqual(at.progress[0].proto.value, 100)
        self.clear_prediction_resources()


if __name__ == "__main__":
    unittest.main()
