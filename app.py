from __future__ import annotations

import io
import math

import streamlit as st
from PIL import Image, UnidentifiedImageError


@st.cache_resource
def _prediction_api():
    from src.predict import predict

    return predict


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
            if st.button("Predict", type="primary"):
                try:
                    class_name, confidence = _prediction_api()(uploaded_image)
                    if (
                        not isinstance(class_name, str)
                        or not class_name.strip()
                        or not isinstance(confidence, (int, float))
                        or not math.isfinite(confidence)
                        or not 0.0 <= confidence <= 1.0
                    ):
                        raise ValueError("Prediction API returned an invalid result.")
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


if __name__ == "__main__":
    main()
