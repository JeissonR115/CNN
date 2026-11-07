# Proposal

## Why

El proyecto necesita una entrega que otra persona pueda instalar, entrenar y ejecutar sin depender de instrucciones dispersas ni de archivos locales no versionados. Un manifiesto compatible, una guía de uso y exclusiones explícitas hacen reproducible el flujo y reducen el riesgo de publicar credenciales o datos pesados.

## What Changes

- Definir `requirements.txt` para Python 3.10 o superior, incluyendo dependencias necesarias para el entrenamiento, evaluación y aplicación Streamlit.
- Completar `README.md` con la estructura final del repositorio y pasos de instalación, preparación del dataset, entrenamiento y ejecución de la aplicación.
- Añadir o completar `.gitignore` para excluir entornos locales, datasets, credenciales, cachés y artefactos generados que no deban versionarse.

## Capabilities

### New Capabilities
- `repository-delivery`: requisitos verificables para instalar, ejecutar y entregar el proyecto con documentación y exclusiones de archivos locales.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta `requirements.txt`, `README.md` y `.gitignore`.
- Documenta el flujo existente o planificado por `src/train.py`, `src/predict.py`, `app.py`, el notebook y los artefactos de `models/`.
- No cambia la arquitectura, el comportamiento de entrenamiento ni la API de predicción.
