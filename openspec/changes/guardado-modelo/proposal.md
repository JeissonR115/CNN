# Proposal

## Why

La aplicación de inferencia necesita cargar el mismo clasificador seleccionado durante el entrenamiento y traducir sus índices de salida a nombres de clase sin ambigüedad. Guardar y verificar conjuntamente el modelo y el orden de clases evita que un artefacto incompleto o incompatible produzca predicciones mal etiquetadas.

## What Changes

- Guardar el mejor modelo entrenado en formato Keras nativo `.keras`.
- Guardar los nombres de clase en `class_names.json` en el mismo orden que las salidas softmax del modelo.
- Verificar que ambos artefactos existen, que el modelo puede volver a cargarse y que sus salidas siguen siendo compatibles con las clases registradas.

## Capabilities

### New Capabilities
- `model-artifacts`: persistencia y verificación de los artefactos necesarios para cargar el clasificador y decodificar sus clases.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta al guardado de `ModelCheckpoint` en `src/model.py` o en el flujo de entrenamiento previsto en `notebooks/entrenamiento.ipynb`, y a la carga posterior desde Streamlit.
- Usa TensorFlow/Keras para el formato `.keras` y JSON estándar para los nombres de clase; no requiere dependencias nuevas.
- Se propone guardar `models/cnn_classifier.keras` y `models/class_names.json`, respetando la ubicación `models/` prevista en `PLAN_INGENIERIA.md`.
