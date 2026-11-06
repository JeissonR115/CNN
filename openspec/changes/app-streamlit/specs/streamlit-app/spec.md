# Spec Delta

## Purpose

Permite que una persona pruebe el clasificador con una imagen cargada y entienda tanto la clase predicha como la confianza asociada a esa predicción.

## ADDED Requirements

### Requirement: The user can provide an image for classification
The application SHALL allow a user to select a JPEG or PNG image and submit it for classification.

#### Scenario: Submitting a supported image
- **WHEN** the user selects and submits a readable JPEG or PNG image
- **THEN** the application passes that image to the project's shared prediction API

#### Scenario: Submitting an unsupported or unreadable image
- **WHEN** the selected file is not a readable supported image
- **THEN** the application displays a clear error and does not display a prediction

### Requirement: The application displays the predicted class and confidence
After a successful prediction, the application SHALL display the predicted class and its confidence as a numeric value and a visual bar scaled to the inclusive range `[0, 1]`.

#### Scenario: Prediction succeeds
- **WHEN** the shared prediction API returns a class and confidence
- **THEN** the application displays that class and confidence as a percentage with a corresponding probability bar

### Requirement: The application reports prediction failures
The application SHALL communicate prediction and required-artifact failures to the user without presenting a fabricated result.

#### Scenario: Prediction resources are unavailable
- **WHEN** the shared prediction API cannot load or validate the required model artifacts
- **THEN** the application displays a clear error and does not display a class or confidence
