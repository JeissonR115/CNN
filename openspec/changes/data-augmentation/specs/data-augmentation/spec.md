# Spec Delta

## Purpose

Define controlled, inspectable image augmentation for model training so the training data gains visual variety while validation and test remain unchanged for fair evaluation.

## ADDED Requirements

### Requirement: Augmentation is restricted to training data
The data pipeline SHALL apply image augmentation only to the training partition after the train/validation/test split, without modifying source images or applying augmentation to validation or test.

#### Scenario: Training samples are augmented
- **WHEN** a training batch is prepared
- **THEN** its images may receive the configured random transformations while the source files remain unchanged

#### Scenario: Validation and test samples are prepared
- **WHEN** a validation or test batch is prepared
- **THEN** each image receives only the standard preprocessing and no random augmentation

### Requirement: Training augmentation uses the specified transformations
The training pipeline SHALL support horizontal flip with probability 0.5, rotation angles from -15° to +15°, zoom scales from 0.9 to 1.1, and contrast factors from 0.8 to 1.2.

#### Scenario: Applying training transformations
- **WHEN** augmentation is applied to a training image
- **THEN** each enabled transformation uses its specified range and no vertical flip is introduced

#### Scenario: Augmented pixel values remain valid
- **WHEN** transformations are applied to an image normalized to `[0, 1]`
- **THEN** the resulting image remains finite and within `[0, 1]`

### Requirement: Augmentation examples are visualized
The project SHALL provide a labeled visualization that compares original training images with examples of the configured transformations, including flip, rotation, zoom, contrast, and a combined augmented result, for each class with training samples.

#### Scenario: Reviewing augmentation examples
- **WHEN** the augmentation preview is generated from readable training images
- **THEN** it displays at least one original training image per represented class beside labeled examples of each configured effect and the combined pipeline

#### Scenario: Repeating the preview
- **WHEN** the preview is generated again with the same sample-selection and visualization seeds
- **THEN** it selects the same source images and produces the same preview transformations
