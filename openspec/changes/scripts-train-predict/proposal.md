# Proposal

## Why

El proyecto necesita flujos reutilizables fuera del notebook para entrenar el clasificador y usarlo desde Streamlit sin duplicar la lógica de inferencia. Mantener además el notebook de trabajo dentro del repositorio facilita reproducir, revisar y entregar el proceso completo.

## What Changes

- Añadir `src/train.py` como punto de entrada para cargar el dataset, ejecutar el flujo de entrenamiento acordado y guardar el modelo y los nombres de clase.
- Añadir `src/predict.py` con la API `predict(image) -> (clase, confianza)`, aceptando una imagen `PIL.Image.Image` y devolviendo la etiqueta predicha y su probabilidad.
- Añadir y mantener versionado `notebooks/entrenamiento.ipynb` como recorrido reproducible que usa los módulos del proyecto y documenta los resultados.
- Hacer que la inferencia reutilice el modelo `.keras`, el `class_names.json` y el preprocesamiento acordado, e informe errores de carga o entrada explícitamente.

## Capabilities

### New Capabilities
- `training-pipeline`: ejecución del entrenamiento mediante un script del repositorio y notebook reproducible mantenido en Git.
- `image-prediction`: API reutilizable para clasificar imágenes PIL y devolver etiqueta y confianza con los artefactos guardados.

### Modified Capabilities
- Ninguna.

## Impact

- Añade `src/train.py`, `src/predict.py` y `notebooks/entrenamiento.ipynb`; integra los módulos propuestos en los cambios `preprocesamiento-division`, `data-augmentation`, `modelo-cnn`, `evaluacion` y `guardado-modelo`.
- `predict(image)` recibe `PIL.Image.Image` en RGB o convertible a RGB y retorna `(str, float)`, donde la confianza es la probabilidad softmax de la clase predicha.
- Reutiliza TensorFlow/Keras, Pillow y dependencias ya contempladas en `PLAN_INGENIERIA.md`; no requiere nuevos paquetes.
