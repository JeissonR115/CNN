from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import preprocess_image
from src.model import CLASS_NAMES_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def _load_model_and_classes(
    model_path: str, class_names_path: str
) -> tuple[object, tuple[str, ...]]:
    import tensorflow as tf

    model_file = Path(model_path)
    names_file = Path(class_names_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model artifact is missing: {model_file}.")
    if not names_file.is_file():
        raise FileNotFoundError(f"Class-name artifact is missing: {names_file}.")
    try:
        model = tf.keras.models.load_model(model_file)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Could not load model artifact {model_file}.") from error
    try:
        class_names = json.loads(names_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read class-name artifact {names_file} as JSON."
        ) from error

    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError(
            "Class-name artifact must be a non-empty JSON array of unique, "
            "non-empty strings."
        )
    output_count = model.output_shape[-1]
    if output_count is None or int(output_count) != len(class_names):
        raise ValueError(
            "Model output count does not match the class-name artifact: "
            f"{output_count} != {len(class_names)}."
        )
    return model, tuple(class_names)


def clear_prediction_cache() -> None:
    _load_model_and_classes.cache_clear()


def load_prediction_resource() -> tuple[object, tuple[str, ...]]:
    return _load_model_and_classes(
        str(MODEL_PATH.resolve()), str(CLASS_NAMES_PATH.resolve())
    )


def predict(image: Image.Image) -> tuple[str, float]:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a readable PIL.Image.Image instance.")
    pixels = preprocess_image(image)
    model, class_names = load_prediction_resource()
    output = model(np.expand_dims(pixels, axis=0), training=False)
    if hasattr(output, "numpy"):
        output = output.numpy()
    probabilities = np.asarray(output, dtype=np.float32)
    if probabilities.shape != (1, len(class_names)):
        raise ValueError(
            "Model returned an incompatible prediction shape: "
            f"{probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite class probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Model probabilities must be in the range [0, 1].")

    predicted_index = int(np.argmax(probabilities[0]))
    return class_names[predicted_index], float(probabilities[0, predicted_index])
