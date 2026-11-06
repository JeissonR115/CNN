# Spec Delta

## Purpose

Define a reproducible exploration of the selected image-classification dataset so its source, class balance, image dimensions, and visual contents can be verified before model training.

## ADDED Requirements

### Requirement: Dataset selection is documented and validated
The exploration SHALL identify the dataset source, its observed class labels and available partitions, and SHALL report whether the dataset has between 3 and 10 non-empty classes.

#### Scenario: Candidate dataset meets the class range
- **WHEN** the Intel Image Classification dataset is available and its directory structure is read
- **THEN** the report identifies the Kaggle source, lists the observed classes and partitions, and confirms whether the observed class count is within the inclusive range of 3 to 10

#### Scenario: Dataset is unavailable or outside the class range
- **WHEN** the dataset cannot be accessed, has no readable class labels, or has fewer than 3 or more than 10 non-empty classes
- **THEN** the exploration reports the blocking condition and does not present the dataset as validated for training

### Requirement: Class distribution is reported
The exploration SHALL report exact readable-image counts for each observed class within each available partition and totals that can be reconciled with those counts.

#### Scenario: All partitions contain readable images
- **WHEN** the exploration completes for a dataset with readable images
- **THEN** each partition's report lists every observed class and its image count, and the partition and dataset totals equal the sums of their class counts

#### Scenario: Images cannot be decoded
- **WHEN** one or more candidate image files cannot be decoded
- **THEN** the report identifies the excluded files or their count and the class-distribution totals count only readable images

### Requirement: Image dimensions are summarized from files
The exploration SHALL measure image width and height from readable image files and report the observed dimensions and the number of images for each dimension.

#### Scenario: Dataset has a consistent image size
- **WHEN** every readable image has the same width and height
- **THEN** the report states that dimension pair and the number of images measured

#### Scenario: Dataset has multiple image sizes
- **WHEN** readable images have more than one width-and-height pair
- **THEN** the report lists each observed dimension pair with its count rather than presenting one approximate size as universal

### Requirement: A labeled example grid is produced
The exploration SHALL produce a visual grid of readable examples labeled with their observed classes, using a reproducible selection when more than one example is available for a class.

#### Scenario: Every observed class has a readable example
- **WHEN** at least one readable image exists for each non-empty class
- **THEN** the grid displays at least one example for every observed class with its class label

#### Scenario: A class has no readable example
- **WHEN** an observed class has no readable image available for the grid
- **THEN** the report identifies that class as missing an example instead of labeling an image with an unsupported class
