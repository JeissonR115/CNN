from __future__ import annotations

import io
import math

import streamlit as st
from PIL import Image, UnidentifiedImageError


@st.cache_resource
def _prediction_api():
    from src.predict import load_prediction_resource, predict_all

    load_prediction_resource()
    return predict_all


def decode_uploaded_image(contents: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ValueError("The uploaded file is not a readable JPEG or PNG image.") from error


def main() -> None:
    st.set_page_config(page_title="CNN image classifier", page_icon="🖼️")
    st.title("CNN image classifier")
    st.write("Upload a JPEG or PNG image to classify it.")
    st.caption(
        "Train the model first to create `models/cnn_classifier.keras` and "
        "`models/class_names.json`."
    )

    try:
        predictor = _prediction_api()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        predictor = None
        st.error(f"Could not load model artifacts: {error}")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        try:
            uploaded_image = decode_uploaded_image(uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))
        else:
            st.image(uploaded_image, caption="Uploaded image")
            if st.button("Predict", type="primary", disabled=predictor is None):
                try:
                    predictions = predictor(uploaded_image)
                    if not predictions:
                        raise ValueError("Prediction API returned an invalid result.")
                    class_name, confidence = predictions[0]
                    if (
                        any(
                            not isinstance(name, str)
                            or not name.strip()
                            or not isinstance(probability, (int, float))
                            or not math.isfinite(probability)
                            or not 0.0 <= probability <= 1.0
                            for name, probability in predictions
                        )
                        or len({name for name, _ in predictions}) != len(predictions)
                        or any(
                            predictions[index][1] < predictions[index + 1][1]
                            for index in range(len(predictions) - 1)
                        )
                    ):
                        raise ValueError("Prediction API returned invalid probabilities.")
                except (
                    FileNotFoundError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    st.error(f"Could not classify the uploaded image: {error}")
                else:
                    st.subheader(f"Predicted class: {class_name}")
                    st.write(f"Confidence: {confidence:.1%}")
                    st.progress(float(confidence), text="Predicted-class probability")
                    st.subheader("Probabilities by class")
                    import matplotlib.pyplot as plt

                    figure, axis = plt.subplots(figsize=(8, 4))
                    axis.bar(
                        [name for name, _ in predictions],
                        [probability for _, probability in predictions],
                    )
                    axis.set(
                        xlabel="Class",
                        ylabel="Probability",
                        ylim=(0.0, 1.0),
                    )
                    axis.tick_params(axis="x", labelrotation=45)
                    figure.tight_layout()
                    st.pyplot(figure)
                    plt.close(figure)


if __name__ == "__main__":
    main()
