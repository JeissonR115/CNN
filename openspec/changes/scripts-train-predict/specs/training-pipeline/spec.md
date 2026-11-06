# Spec Delta

## Purpose

Provide a repository-owned entry point for training the image classifier and a checked-in notebook that documents and demonstrates the project's reproducible workflow.

## ADDED Requirements

### Requirement: Training is runnable from a repository script
The project SHALL provide `src/train.py` as a runnable training entry point that accepts the dataset root explicitly and orchestrates the agreed 128 × 128 preprocessing, stratified 70/15/15 split, train-only augmentation, CNN training, and saved-artifact verification.

#### Scenario: Running training with a dataset root
- **WHEN** a user invokes the training entry point with a valid dataset root
- **THEN** it runs the configured training flow and produces the verified model and class-name artifacts at the documented paths

#### Scenario: Dataset root is unavailable or invalid
- **WHEN** the supplied dataset root does not exist or has no usable labeled images
- **THEN** training reports the input problem explicitly and does not report successful completion

### Requirement: The project notebook is maintained in the repository
The project SHALL include `notebooks/entrenamiento.ipynb` in the repository as a readable, executable walkthrough that uses the project training and prediction modules rather than depending on notebook-only implementations.

#### Scenario: Reviewing the repository notebook
- **WHEN** the repository is checked out or its notebook is opened
- **THEN** `notebooks/entrenamiento.ipynb` is present and includes the documented workflow for dataset exploration, augmentation preview, training, evaluation, and prediction using repository modules

#### Scenario: Executing notebook cells from the repository
- **WHEN** the documented prerequisites and dataset path are available
- **THEN** the notebook cells can run in order from the repository without relying on an untracked or Colab-temporary copy of project code
