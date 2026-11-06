# Spec Delta

## Purpose

Define a final, reproducible evaluation of the selected image classifier using held-out test data, with aggregate metrics, training-history plots, and individual predictions that make errors and confidence inspectable.

## ADDED Requirements

### Requirement: Final metrics use the held-out test set
The evaluation SHALL report test accuracy, per-class precision, recall and F1-score with support, and a confusion matrix using the same ordered class labels as the dataset.

#### Scenario: Evaluating a trained classifier
- **WHEN** a selected model and a non-empty labeled test set are available
- **THEN** the report displays accuracy, a classification report with per-class metrics and support, and a confusion matrix whose rows are true classes and columns are predicted classes

#### Scenario: Test set is empty
- **WHEN** evaluation is requested with no labeled test examples
- **THEN** the workflow reports that evaluation cannot be performed and does not produce success-shaped metrics

### Requirement: Training and validation history is plotted
The evaluation output SHALL include loss and accuracy plots over epochs for both training and validation using the recorded fit history.

#### Scenario: Plotting complete training history
- **WHEN** fit history contains loss, validation loss, accuracy, and validation accuracy
- **THEN** the output shows epoch-indexed plots that distinguish training from validation for loss and accuracy

#### Scenario: History is missing a required series
- **WHEN** the fit history does not contain one or more required metric series
- **THEN** the workflow identifies the missing series instead of silently omitting the corresponding curve

### Requirement: Individual test predictions are presented
The evaluation SHALL show at least 10 test predictions when 10 or more test examples exist, including each sample's true class, predicted class, and probability for the predicted class, with reproducible sample selection.

#### Scenario: Test set contains at least 10 examples
- **WHEN** the test set has 10 or more labeled examples
- **THEN** the output displays at least 10 distinct samples with true label, predicted label, and predicted-class probability between 0 and 1

#### Scenario: Test set contains fewer than 10 examples
- **WHEN** the test set has between 1 and 9 labeled examples
- **THEN** the output displays every available example and explicitly states that the requested minimum of 10 could not be met

#### Scenario: Repeating sample selection
- **WHEN** evaluation is repeated with the same test set and sample-selection seed
- **THEN** the same test examples are selected for the individual-prediction display

### Requirement: Test data remains isolated from model selection
The evaluation SHALL use test labels only after training and checkpoint selection are complete, and SHALL NOT use test metrics or predictions to tune the model or its training callbacks.

#### Scenario: Evaluating the selected checkpoint
- **WHEN** the best checkpoint selected using validation data is evaluated
- **THEN** test data is used only to calculate and display final metrics and predictions
