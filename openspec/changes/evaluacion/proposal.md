# Proposal

## Why

El proyecto necesita una evaluación final clara y reproducible que permita medir el clasificador en datos no usados para entrenar ni seleccionar el checkpoint. Las métricas, gráficas de aprendizaje y ejemplos de predicción aportan evidencia complementaria sobre rendimiento, errores y confianza del modelo.

## What Changes

- Evaluar el mejor modelo guardado sobre el conjunto test reservado y reportar accuracy, `classification_report` y matriz de confusión.
- Graficar loss y accuracy de entrenamiento y validación a partir del historial de ajuste.
- Mostrar al menos 10 predicciones de test con clase real, clase predicha y probabilidad de la clase predicha; seleccionar muestras de forma reproducible.
- Si test contiene menos de 10 ejemplos, mostrar todos e indicar que no se alcanzó el mínimo solicitado.
- Mantener test fuera del ajuste, selección de checkpoint y callbacks.

## Capabilities

### New Capabilities
- `model-evaluation`: evaluación final del clasificador, visualizaciones de entrenamiento y revisión de predicciones individuales.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta al flujo de evaluación previsto en `notebooks/entrenamiento.ipynb`, que todavía no existe.
- Se apoya en TensorFlow/Keras, scikit-learn y Matplotlib, ya contemplados en `PLAN_INGENIERIA.md`; no requiere nuevas dependencias.
- Consume el checkpoint de mejor `val_loss` producido por `modelo-cnn` y el conjunto test etiquetado definido por `preprocesamiento-division`.
