# repository-delivery Specification

## Purpose

Define una entrega reproducible del proyecto que permita instalar dependencias, preparar el dataset, entrenar el clasificador y ejecutar la interfaz, a la vez que evita versionar archivos locales sensibles o voluminosos.

## Requirements

### Requirement: The project declares a supported Python baseline and dependencies
The repository SHALL declare Python 3.10 or later and list the dependencies required for training, evaluation, image prediction, and the Streamlit application in its installable dependency manifest.

#### Scenario: Installing project dependencies with a supported interpreter
- **WHEN** a user uses Python 3.10 or later and installs the declared dependencies
- **THEN** the environment can import the packages required by the documented training and application workflows

#### Scenario: Using an older interpreter
- **WHEN** a user attempts to follow the installation instructions with Python earlier than 3.10
- **THEN** the documented Python requirement clearly identifies that interpreter as unsupported

### Requirement: The README documents the reproducible project workflow
The repository SHALL provide a README with the final project structure and the steps needed to install dependencies, prepare the dataset, train the model, and launch the application.

#### Scenario: Following the local workflow
- **WHEN** a user follows the README from a clean checkout with a supported Python version and dataset
- **THEN** the instructions identify the commands to install dependencies, train with the dataset path, and start the Streamlit application with its required model artifacts

#### Scenario: Reviewing the repository layout
- **WHEN** a user consults the README
- **THEN** it shows the final locations of the application, source modules, notebook, model artifacts, and project documentation

### Requirement: Local secrets and generated data are excluded from version control
The repository SHALL ignore local credentials, downloaded datasets, environment directories, caches, and generated model artifacts that are not intended to be distributed in Git.

#### Scenario: Checking ignored local files
- **WHEN** the ignore rules are checked against representative Kaggle credentials, dataset files, local environments, caches, and generated model artifacts
- **THEN** each is reported as ignored and none needs to be added to the repository to run the documented workflow
