# Spec Delta

## Purpose

Define a consistent image transformation and a reproducible, class-stratified split of labeled data so model training, validation, testing, and inference use compatible inputs.

## ADDED Requirements

### Requirement: Images are resized to the model input dimensions
The preprocessing flow SHALL convert each readable labeled image to RGB and resize it to exactly 128 × 128 pixels before providing it to the model.

#### Scenario: Processing an image of another size or mode
- **WHEN** a readable labeled image has dimensions other than 128 × 128 or is not encoded as RGB
- **THEN** the model input has dimensions 128 × 128 and three RGB channels

### Requirement: RGB values are normalized consistently
The preprocessing flow SHALL convert 8-bit RGB pixel values to floating-point values in the inclusive range `[0, 1]` by scaling each channel by 1/255.

#### Scenario: Normalizing minimum and maximum channel values
- **WHEN** an image containing channel values 0 and 255 is preprocessed
- **THEN** the resulting corresponding values are 0.0 and 1.0, respectively

### Requirement: Labeled data is split by class
The data preparation flow SHALL divide the readable labeled image set into mutually exclusive training, validation, and test subsets with target proportions of 70 %, 15 %, and 15 %, preserving class representation as closely as integer sample counts allow.

#### Scenario: Stratified division of adequately sized classes
- **WHEN** every class has enough readable labeled examples to be represented in all three subsets
- **THEN** each subset contains examples from every class and its sample counts approximate the target proportions, with integer rounding accounted for

#### Scenario: A class has too few examples
- **WHEN** a class has too few readable labeled examples to preserve its representation across all three subsets
- **THEN** the flow reports that stratified division cannot satisfy the requirement and does not silently fall back to a non-stratified split

#### Scenario: Dataset includes an unlabeled official test partition
- **WHEN** the candidate dataset includes test files without class labels
- **THEN** those files are excluded from the 70/15/15 split, which is performed only on readable labeled images

### Requirement: The split is reproducible and has no leakage
The flow SHALL use split seed 42 and assign the same labeled inputs to the same subsets on repeated runs; no image SHALL occur in more than one subset.

#### Scenario: Repeating the split
- **WHEN** the split is performed more than once with unchanged inputs
- **THEN** the image membership of training, validation, and test is identical in each run

#### Scenario: Checking subset membership
- **WHEN** the training, validation, and test subsets are produced
- **THEN** their image paths are pairwise disjoint

### Requirement: Preprocessing is consistent at inference
The inference flow SHALL use the same RGB conversion, 128 × 128 resize, and `[0, 1]` scaling as the training input pipeline.

#### Scenario: Applying preprocessing during inference
- **WHEN** an image is prepared for prediction
- **THEN** its model input has the same channel order, dimensions, data type, and value range as a preprocessed training image
