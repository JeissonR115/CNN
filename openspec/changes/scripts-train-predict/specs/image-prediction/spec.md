# Spec Delta

## Purpose

Expose a reusable image-classification API that maps a PIL image to a dataset class and its predicted probability using the saved project model.

## ADDED Requirements

### Requirement: Prediction returns a class and confidence
The project SHALL provide `src/predict.py` with `predict(image) -> (class_name, confidence)`, loading `models/cnn_classifier.keras` and `models/class_names.json`, accepting a `PIL.Image.Image`, and returning a non-empty class-name string and the predicted class probability as a float in `[0, 1]`.

#### Scenario: Predicting a supported image
- **WHEN** a readable PIL image and valid model artifacts are available
- **THEN** `predict(image)` returns the class name corresponding to the highest model output and that output's probability

### Requirement: Prediction uses the training input preprocessing
The prediction API SHALL convert the input image to RGB, resize it to 128 × 128, and scale 8-bit channel values to float values in `[0, 1]` before model inference.

#### Scenario: Predicting from a different image mode or size
- **WHEN** a readable PIL image is not RGB or is not 128 × 128
- **THEN** prediction converts and resizes it to the model's required RGB input shape and scales its values consistently with training

### Requirement: Prediction validates artifacts and image input
The prediction API SHALL report errors explicitly when the input is not a readable PIL image, either model artifact is missing or invalid, or the model output dimension does not match the saved class-name list.

#### Scenario: Required model artifact is absent
- **WHEN** the model file or class-name metadata cannot be loaded
- **THEN** prediction raises or surfaces a descriptive error and does not return a fabricated class or confidence

#### Scenario: Model output and class names disagree
- **WHEN** the loaded model output dimension differs from the number of saved class names
- **THEN** prediction reports the incompatibility and does not map an output index to a class

#### Scenario: Input is not a usable PIL image
- **WHEN** the caller supplies an object that is not a readable `PIL.Image.Image`
- **THEN** prediction reports the invalid input instead of silently substituting an image

### Requirement: The application delegates image inference to the shared API
The Streamlit application SHALL use the shared prediction API for image preprocessing, model inference, class-name mapping, and confidence instead of maintaining a separate inference implementation.

#### Scenario: Predicting through the application
- **WHEN** a user submits an uploaded image in Streamlit
- **THEN** the application displays the class and confidence returned by the shared prediction API
