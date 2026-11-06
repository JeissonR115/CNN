# Proposal

## Why

El entrenamiento requiere que todas las imágenes tengan una forma de entrada común y una escala numérica consistente, y que sus conjuntos de entrenamiento, validación y prueba sean comparables por clase. Definir estas transformaciones y una división reproducible evita inconsistencias entre entrenamiento y evaluación.

## What Changes

- Redimensionar las imágenes etiquetadas a 128 × 128 píxeles.
- Normalizar los valores RGB de 8 bits al rango `[0, 1]`.
- Dividir estratificadamente las imágenes etiquetadas en 70 % para entrenamiento, 15 % para validación y 15 % para prueba.
- Mantener la asignación de muestras reproducible con semilla fija 42 y evitar solapamientos entre particiones.
- Excluir de la división los archivos del test oficial de Kaggle que no tengan etiquetas.

## Capabilities

### New Capabilities
- `dataset-preprocessing`: preparación reproducible de imágenes y división estratificada de datos etiquetados para entrenamiento, validación y prueba.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta al flujo de datos previsto en `notebooks/entrenamiento.ipynb` y a la consistencia del preprocesamiento durante inferencia en Streamlit.
- El repositorio todavía no contiene notebook ni código de entrenamiento; la implementación deberá integrarse en el flujo previsto.
- La división 70/15/15 se aplica al conjunto de imágenes etiquetadas, no a archivos de test sin etiquetas. Esta decisión precisa el plan de ingeniería, que proponía reservar una partición oficial de prueba.
- No se requieren nuevas dependencias según el stack descrito en `PLAN_INGENIERIA.md`.
