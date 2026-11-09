from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.data import IMAGE_SIZE

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_classifier.keras"
CLASS_NAMES_PATH = MODEL_PATH.with_name("class_names.json")
MAX_EPOCHS = 30
DEFAULT_EPOCHS = 2
BATCH_SIZE = 32


def build_classifier(num_classes: int):
    import tensorflow as tf

    if not 3 <= num_classes <= 10:
        raise ValueError("num_classes must be between 3 and 10.")

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), name="image")
    values = inputs
    for block, filters in enumerate((32, 64, 128), start=1):
        values = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name=f"block{block}_conv",
        )(values)
        values = tf.keras.layers.MaxPooling2D(
            pool_size=2, name=f"block{block}_pool"
        )(values)
    values = tf.keras.layers.Flatten(name="flatten")(values)
    values = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(values)
    values = tf.keras.layers.Dropout(0.5, name="dense_dropout")(values)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(values)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")


def compile_classifier(model: Any) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_training_callbacks(
    checkpoint_path: str | Path = MODEL_PATH,
) -> list[Any]:
    import tensorflow as tf

    destination = Path(checkpoint_path)
    if destination.suffix != ".keras":
        raise ValueError("The model checkpoint path must end in '.keras'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(destination),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.000001,
        ),
    ]


def fit_classifier(
    model: Any,
    training_data: Any,
    validation_data: Any,
    *,
    checkpoint_path: str | Path = MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
) -> Any:
    if validation_data is None:
        raise ValueError("Validation data is required because callbacks monitor val_loss.")
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}.")

    compile_classifier(model)
    model.summary()
    return model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=make_training_callbacks(checkpoint_path),
    )


def save_and_verify_artifacts(
    model: Any,
    class_names: list[str] | tuple[str, ...],
    verification_sample: Any,
    *,
    model_path: str | Path = MODEL_PATH,
    class_names_path: str | Path = CLASS_NAMES_PATH,
) -> None:
    import json

    import numpy as np
    import tensorflow as tf

    destination = Path(model_path)
    names_path = Path(class_names_path)
    names = list(class_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Class names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique.")
    if verification_sample is None:
        raise ValueError("A labeled, non-test sample is required to verify artifacts.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    names_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        raise FileNotFoundError(
            f"Best model checkpoint was not created: {destination}."
        )

    try:
        reloaded = tf.keras.models.load_model(destination)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Could not reload model artifact {destination}.") from error
    output_count = reloaded.output_shape[-1]
    if output_count is None or int(output_count) != len(names):
        raise ValueError(
            "Model output count does not match the saved class-name count: "
            f"{output_count} != {len(names)}."
        )
    temporary_names_path = names_path.with_name(f".{names_path.name}.tmp")
    try:
        temporary_names_path.write_text(
            json.dumps(names, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loaded_names = json.loads(
            temporary_names_path.read_text(encoding="utf-8")
        )
        if loaded_names != names:
            raise ValueError("Saved class names do not match the training order.")
        os.replace(temporary_names_path, names_path)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"Could not save and reload class-name artifact {names_path}."
        ) from error
    finally:
        temporary_names_path.unlink(missing_ok=True)

    try:
        np.testing.assert_allclose(
            model(verification_sample, training=False).numpy(),
            reloaded(verification_sample, training=False).numpy(),
            atol=1e-6,
            rtol=1e-5,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Reloaded model predictions differ from the trained model."
        ) from error
