# Tasks

## 1. Dependency manifest

- [ ] 1.1 Create `requirements.txt` with the Python 3.10 minimum and compatible direct dependencies for training, evaluation, prediction, and Streamlit; verify installation in a supported environment, package imports, and dependency consistency.

## 2. Usage and repository documentation

- [ ] 2.1 Update `README.md` with the verified Python compatibility range, environment setup and dependency installation; verify every documented prerequisite matches the manifest and supported runtimes.
- [ ] 2.2 Document dataset preparation, the `python -m src.train --data-dir <ruta>` command, required generated model artifacts, `streamlit run app.py`, notebook usage, and the final repository structure; verify referenced paths exist and commands match the implemented entry points.

## 3. Version-control exclusions

- [ ] 3.1 Create or update `.gitignore` for local environments, Kaggle credentials, downloaded datasets, caches, notebook outputs, and generated model artifacts while retaining required source files and placeholders; verify representative paths with `git check-ignore` and inspect that required project files remain trackable.

## 4. Delivery consistency

- [ ] 4.1 Review the README, dependency manifest, ignore rules, and related training/Streamlit instructions together; verify a clean-checkout workflow is documented without requiring credentials, dataset files, or model artifacts to be committed.
