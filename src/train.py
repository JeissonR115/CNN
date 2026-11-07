from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    load_labeled_records,
    make_dataset,
    split_dataset,
)
from src.model import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    MAX_EPOCHS,
    MODEL_PATH,
    build_classifier,
    fit_classifier,
    save_and_verify_artifacts,
)


def train(
    dataset_root: str | Path,
    *,
    epochs: int = MAX_EPOCHS,
):
    records = load_labeled_records(dataset_root)
    splits = split_dataset(records)
    training_data = make_dataset(
        splits.train,
        splits.class_names,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
    )
    validation_data = make_dataset(
        splits.validation,
        splits.class_names,
        batch_size=BATCH_SIZE,
    )
    model = build_classifier(len(splits.class_names))
    history = fit_classifier(
        model,
        training_data,
        validation_data,
        checkpoint_path=MODEL_PATH,
        epochs=epochs,
    )
    validation_images, _ = next(iter(validation_data))
    save_and_verify_artifacts(
        model,
        splits.class_names,
        validation_images[:1],
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH,
    )
    return model, history, splits.class_names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and verify the image classification CNN."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Dataset root: CIFAR-10 batch directory or labeled image-class folders."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Maximum training epochs (1-{MAX_EPOCHS}; default: {MAX_EPOCHS}).",
    )
    arguments = parser.parse_args(argv)
    try:
        _, history, class_names = train(
            arguments.data_dir, epochs=arguments.epochs
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    completed_epochs = len(history.epoch)
    print(
        f"Training complete: {completed_epochs} epoch(s), "
        f"{len(class_names)} class(es)."
    )
    print(f"Model: {MODEL_PATH}")
    print(f"Class names: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
