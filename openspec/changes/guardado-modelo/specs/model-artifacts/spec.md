# Spec Delta

## Purpose

Define durable model and class-label artifacts that can be loaded together by evaluation and inference without losing the mapping between model outputs and human-readable classes.

## ADDED Requirements

### Requirement: The selected model is saved in native Keras format
The training workflow SHALL save the best validation-selected classifier as `models/cnn_classifier.keras` in native Keras format.

#### Scenario: Training completes with a best checkpoint
- **WHEN** training finishes after selecting checkpoints by validation loss
- **THEN** `models/cnn_classifier.keras` exists and contains the best selected model in native Keras format

### Requirement: Class names are saved in output order
The training workflow SHALL save `models/class_names.json` as a JSON array of unique class-name strings ordered so each array index corresponds to the same-index model output.

#### Scenario: Saving class metadata
- **WHEN** the trained model has an output for each configured class
- **THEN** `class_names.json` contains exactly one unique non-empty name per output in the training label order

### Requirement: Saved artifacts are verified by reloading
Before declaring the model artifacts ready, the workflow SHALL load the saved model and class names again and verify that the model output dimension equals the number of class names and that reloaded predictions match the pre-save model predictions for the same sample within numerical tolerance.

#### Scenario: Reloading a compatible model and class list
- **WHEN** the saved model and class metadata are reloaded successfully
- **THEN** the class count matches the model output dimension and predictions on the same input are numerically equivalent within a documented tolerance

#### Scenario: Artifact load or compatibility check fails
- **WHEN** either artifact cannot be loaded or the class count or prediction comparison does not match
- **THEN** the workflow reports the failure and does not present the artifacts as verified
