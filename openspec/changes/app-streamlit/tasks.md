# Tasks

## 1. Runtime setup

- [x] 1.1 Declare Streamlit as an application dependency in the repository's runtime dependency manifest; verify the manifest resolves and Streamlit can be imported in the project environment.

## 2. Image input and prediction UI

- [x] 2.1 Implement the `app.py` image uploader for JPEG and PNG and pass decoded PIL images to `src.predict.predict(image)` without duplicating preprocessing; verify supported formats reach the predictor and unreadable files show an error without invoking it.
- [x] 2.2 Use `@st.cache_resource` for the shared prediction resource, then display the returned class and confidence percentage with a `[0, 1]` probability bar; verify with focused Streamlit tests that the API is reused and its result is rendered accurately.
- [x] 2.3 Surface predictor and model-artifact failures without showing a fabricated prediction; verify missing/incompatible artifacts produce a visible error and no result.

## 3. Usage documentation and integration

- [x] 3.1 Document installation, the `streamlit run app.py` command, the required model artifacts, and the need to restart after replacing cached artifacts; verify each documented path and command matches the implemented app.
- [x] 3.2 Run a focused integration check with the shared `src.predict` API and valid model artifacts; verify an uploaded image is preprocessed by the shared predictor and the returned class/confidence are displayed.
