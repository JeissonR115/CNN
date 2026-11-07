from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.model import (
    build_classifier,
    compile_classifier,
    fit_classifier,
    make_training_callbacks,
    save_and_verify_artifacts,
)


class ClassifierTests(unittest.TestCase):
    def test_classifier_architecture_and_probability_outputs(self) -> None:
        import tensorflow as tf

        for class_count in (6, 4):
            with self.subTest(class_count=class_count):
                model = build_classifier(class_count)
                probabilities = model(
                    tf.zeros((1, 128, 128, 3), dtype=tf.float32), training=False
                )
                self.assertEqual(model.input_shape, (None, 128, 128, 3))
                self.assertEqual(model.output_shape, (None, class_count))
                self.assertAlmostEqual(float(tf.reduce_sum(probabilities[0])), 1.0, 5)
                self.assertEqual(
                    len(
                        [
                            layer
                            for layer in model.layers
                            if isinstance(layer, tf.keras.layers.Conv2D)
                        ]
                    ),
                    6,
                )
                self.assertEqual(
                    len(
                        [
                            layer
                            for layer in model.layers
                            if isinstance(
                                layer, tf.keras.layers.BatchNormalization
                            )
                        ]
                    ),
                    6,
                )
                self.assertEqual(
                    len(
                        [
                            layer
                            for layer in model.layers
                            if isinstance(layer, tf.keras.layers.MaxPooling2D)
                        ]
                    ),
                    3,
                )
                self.assertEqual(model.get_layer("block1_dropout").rate, 0.25)
                self.assertEqual(model.get_layer("dense_dropout").rate, 0.5)

    def test_compile_and_callback_configuration(self) -> None:
        import tensorflow as tf

        model = build_classifier(3)
        compile_classifier(model)
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.001)
        self.assertEqual(model.loss, "sparse_categorical_crossentropy")
        self.assertEqual(model.get_compile_config()["metrics"], ["accuracy"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.keras"
            early_stopping, checkpoint, reduce_lr = make_training_callbacks(
                checkpoint_path
            )
            self.assertEqual(
                (
                    early_stopping.monitor,
                    early_stopping.patience,
                    early_stopping.restore_best_weights,
                ),
                ("val_loss", 5, True),
            )
            self.assertEqual(checkpoint.monitor, "val_loss")
            self.assertEqual(checkpoint.mode, "min")
            self.assertTrue(checkpoint.save_best_only)
            self.assertEqual(
                (
                    reduce_lr.monitor,
                    reduce_lr.factor,
                    reduce_lr.patience,
                    reduce_lr.min_lr,
                ),
                ("val_loss", 0.5, 2, 0.000001),
            )

    def test_callbacks_reduce_lr_restore_best_weights_and_keep_best_checkpoint(self) -> None:
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(2,)),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.keras"
            early_stopping, checkpoint, reduce_lr = make_training_callbacks(
                checkpoint_path
            )
            early_stopping.set_model(model)
            checkpoint.set_model(model)
            reduce_lr.set_model(model)

            early_stopping.on_train_begin()
            checkpoint.on_train_begin()
            reduce_lr.on_train_begin()
            model.set_weights([np.zeros((2, 3)), np.zeros(3)])
            for callback in (early_stopping, checkpoint, reduce_lr):
                callback.on_epoch_end(0, {"val_loss": 1.0})
            best_weights = model.get_weights()

            model.set_weights([np.ones((2, 3)), np.ones(3)])
            for epoch in range(1, 6):
                for callback in (early_stopping, checkpoint, reduce_lr):
                    callback.on_epoch_end(epoch, {"val_loss": 1.0})
            early_stopping.on_train_end()

            self.assertTrue(early_stopping.model.stop_training)
            for expected, actual in zip(best_weights, model.get_weights()):
                np.testing.assert_array_equal(expected, actual)
            self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.0025)
            reloaded = tf.keras.models.load_model(checkpoint_path)
            for expected, actual in zip(best_weights, reloaded.get_weights()):
                np.testing.assert_array_equal(expected, actual)

    def test_training_requires_validation_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "Validation data is required"):
            fit_classifier(build_classifier(3), object(), None)

    def test_best_model_and_class_names_are_reloaded_and_verified(self) -> None:
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = build_classifier(3)
            model_path = root / "cnn_classifier.keras"
            model.save(model_path)
            sample = tf.zeros((1, 128, 128, 3), dtype=tf.float32)

            save_and_verify_artifacts(
                model,
                ("buildings", "forest", "sea"),
                sample,
                model_path=model_path,
                class_names_path=root / "class_names.json",
            )
            with self.assertRaisesRegex(ValueError, "output count"):
                save_and_verify_artifacts(
                    model,
                    ("buildings", "forest"),
                    sample,
                    model_path=model_path,
                    class_names_path=root / "bad_class_names.json",
                )
            self.assertFalse((root / "bad_class_names.json").exists())

    def test_artifact_verification_rejects_missing_and_changed_models(self) -> None:
        import tensorflow as tf

        class DifferentModel:
            output_shape = (None, 3)

            @staticmethod
            def __call__(values, training: bool = False):
                return tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
            model = build_classifier(3)
            missing_path = root / "missing.keras"
            with self.assertRaises(FileNotFoundError):
                save_and_verify_artifacts(
                    model,
                    ("buildings", "forest", "sea"),
                    sample,
                    model_path=missing_path,
                    class_names_path=root / "missing_classes.json",
                )

            model_path = root / "saved.keras"
            model.save(model_path)
            with patch(
                "tensorflow.keras.models.load_model", return_value=DifferentModel()
            ):
                with self.assertRaisesRegex(RuntimeError, "predictions differ"):
                    save_and_verify_artifacts(
                        model,
                        ("buildings", "forest", "sea"),
                        sample,
                        model_path=model_path,
                        class_names_path=root / "different_classes.json",
                    )


if __name__ == "__main__":
    unittest.main()
