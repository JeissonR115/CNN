# Spec Delta

## Purpose

Define an inspectable CNN classifier and a validation-driven training process that produces a model suitable for the project's image classes and preserves its best-performing checkpoint.

## ADDED Requirements

### Requirement: The classifier uses the specified CNN architecture
The model SHALL accept 128 × 128 RGB images and use three convolutional blocks with two Conv2D layers per block, 32, 64, and 128 filters respectively, BatchNormalization after each convolution, MaxPooling and Dropout 0.25 per block, followed by Dense 128, Dropout 0.5, and a softmax output with one unit per class.

#### Scenario: Building the classifier for the candidate dataset
- **WHEN** the model is built for six class labels
- **THEN** it accepts inputs shaped 128 × 128 × 3, outputs six class probabilities summing to approximately 1 for each input, and includes the specified convolutional, normalization, pooling, dropout, dense, and softmax layers

#### Scenario: Building with a different supported class count
- **WHEN** the model is built for a labeled dataset with between 3 and 10 classes
- **THEN** the softmax output dimension matches the configured number of classes

### Requirement: The model architecture is inspectable
The training workflow SHALL print a model summary that reports each layer's output shape and the total, trainable, and non-trainable parameter counts before fitting.

#### Scenario: Reviewing the model before training
- **WHEN** the model is built and training begins
- **THEN** the workflow displays its layer summary and parameter counts before the first training epoch

### Requirement: Training uses the defined optimization settings
The training workflow SHALL compile the model with Adam at learning rate 0.001, sparse categorical crossentropy, and accuracy, and SHALL train with batch size 32 for at most 30 epochs using validation data.

#### Scenario: Starting training with validation data
- **WHEN** fitting begins with labeled training and validation data
- **THEN** the run uses the specified optimizer, learning rate, loss, metric, batch size, and epoch ceiling

#### Scenario: Validation data is unavailable
- **WHEN** fitting is requested without a validation set
- **THEN** the workflow reports the missing validation data and does not start callbacks that depend on `val_loss`

### Requirement: Training stops when validation loss plateaus
The training workflow SHALL monitor `val_loss` with EarlyStopping using patience 5 and restoring the best weights, and ReduceLROnPlateau using factor 0.5, patience 2, and minimum learning rate 0.000001.

#### Scenario: Validation loss stops improving
- **WHEN** `val_loss` does not improve for the configured patience intervals
- **THEN** the learning rate is reduced by the configured factor down to its minimum and training stops after the EarlyStopping patience is exhausted, restoring the best observed weights

### Requirement: The best validation checkpoint is preserved
The training workflow SHALL save only the best model checkpoint according to minimum `val_loss` and SHALL not use test data to select checkpoints or control training.

#### Scenario: Validation loss improves
- **WHEN** an epoch achieves a lower `val_loss` than prior epochs
- **THEN** the saved best checkpoint is updated to that epoch's model

#### Scenario: Test data is available during training
- **WHEN** a separate test set is available
- **THEN** it is not supplied to callbacks or used to select the checkpoint
