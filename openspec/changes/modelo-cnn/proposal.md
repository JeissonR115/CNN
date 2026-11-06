# Proposal

## Why

El proyecto necesita una arquitectura CNN explícita y un flujo de entrenamiento que controle el sobreajuste y conserve el mejor modelo según validación. Definir estas decisiones ahora permite reproducir el entrenamiento en Colab y conectar el artefacto resultante con la inferencia de Streamlit.

## What Changes

- Definir una CNN con tres bloques de dos capas Conv2D (32, 64 y 128 filtros), BatchNormalization, MaxPooling y Dropout.
- Añadir una capa Dense de 128 unidades y dropout antes de una salida softmax dimensionada según el número de clases.
- Exponer `model.summary()` para revisar formas y parámetros entrenables.
- Definir entrenamiento reproducible con Adam, pérdida sparse categorical crossentropy, hasta 30 épocas y batch size 32.
- Configurar EarlyStopping, ModelCheckpoint y ReduceLROnPlateau con `val_loss` como métrica supervisada.

## Capabilities

### New Capabilities
- `cnn-classifier`: arquitectura y entrenamiento del clasificador CNN, incluidos resumen del modelo y control de entrenamiento por validación.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta a `src/model.py` y al flujo de entrenamiento previsto en `notebooks/entrenamiento.ipynb`; estos archivos todavía no existen en el repositorio.
- Usa TensorFlow/Keras, ya contemplado en `PLAN_INGENIERIA.md`; no requiere nuevas dependencias.
- El modelo consume imágenes 128 × 128 RGB normalizadas a `[0, 1]` y el número de clases observado, de acuerdo con los cambios propuestos para preprocesamiento y exploración del dataset.
