# Proposal

## Why

La variación limitada de las imágenes de entrenamiento puede ayudar a que la CNN generalice sin alterar las evaluaciones. El proyecto necesita controlar que estas transformaciones afecten solo a train y poder inspeccionar visualmente sus efectos antes de entrenar.

## What Changes

- Incorporar aumentación moderada de imágenes de entrenamiento mediante flip horizontal, rotación de hasta ±15°, zoom de hasta ±10 % y contraste entre 0,8 y 1,2.
- Aplicar las transformaciones solo a muestras del conjunto train, después de la división, sin modificar imágenes originales ni aumentar validation o test.
- Añadir una visualización reproducible que compare imágenes originales con ejemplos aumentados e identifique los efectos aplicados.

## Capabilities

### New Capabilities
- `data-augmentation`: aumentación visual controlada y exclusiva del entrenamiento, con ejemplos inspeccionables.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta al flujo de entrenamiento previsto en `notebooks/entrenamiento.ipynb`, que todavía no existe en el repositorio.
- Se apoya en TensorFlow/Keras para las transformaciones y Matplotlib para visualizar ejemplos, ambos ya contemplados en `PLAN_INGENIERIA.md`; no requiere nuevas dependencias.
- Las particiones validation y test definidas por `preprocesamiento-division` permanecen sin aumentación.
