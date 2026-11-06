# Design

## Context

El repositorio contiene el plan, pero todavía no tiene módulos de modelo ni notebook de entrenamiento. El plan prevé TensorFlow/Keras, imágenes RGB de 128 × 128 normalizadas a `[0, 1]`, y seis clases para el dataset candidato. Ver `proposal.md` para la motivación y `specs/cnn-classifier/spec.md` para el contrato.

## Goals / Non-Goals

**Goals:**
- Separar la definición del modelo del flujo de entrenamiento para que el notebook pueda inspeccionar, entrenar y guardar el mismo clasificador.
- Controlar el entrenamiento usando exclusivamente el conjunto de validación y conservar el mejor modelo.

**Non-Goals:**
- Transfer learning, búsqueda automática de hiperparámetros o ajuste del umbral de predicción.
- Seleccionar arquitectura o checkpoint según las métricas del conjunto de prueba.
- Fijar una exactitud mínima; el rendimiento depende del entrenamiento y del dataset y se reportará, no se garantizará por la arquitectura.

## Decisions

- Implementar el constructor del modelo en `src/model.py` con bloques secuenciales de dos Conv2D `3 × 3`, padding `same` y ReLU; cada convolución va seguida de BatchNormalization. Cada bloque termina con MaxPooling `2 × 2` y Dropout 0.25. Los filtros por bloque son 32, 64 y 128. Después de los bloques, Flatten, Dense 128 con ReLU, Dropout 0.5 y Dense softmax configurada con la cantidad de clases. Una arquitectura secuencial explícita se prefiere a una CNN funcional más compleja para facilitar su revisión en `model.summary()`.
- Compilar con Adam a learning rate 0.001, sparse categorical crossentropy y accuracy. Las etiquetas enteras evitan exigir codificación one-hot para el dataset.
- Usar batch size 32 y máximo 30 épocas, con `val_loss` como monitor común. EarlyStopping usa paciencia 5 y `restore_best_weights=True`; ReduceLROnPlateau usa factor 0.5, paciencia 2 y mínimo 0.000001; ModelCheckpoint usa `save_best_only=True` y guarda el modelo con menor `val_loss`. La validación es obligatoria porque todos los callbacks se basan en ella.
- Emitir `model.summary()` al construir el modelo, antes de `fit`, y guardar el mejor checkpoint como artefacto de entrenamiento. El test se evalúa después, sin conectarlo a callbacks.

## Risks / Trade-offs

- [La arquitectura fija puede sobreajustar o rendir por debajo de lo esperado] → Usar dropout, early stopping y reportar las métricas de evaluación sin prometer un umbral de exactitud.
- [El tamaño fijo de entrada o las etiquetas no enteras pueden causar incompatibilidades] → Validar forma RGB 128 × 128, número de clases y codificación entera antes de iniciar el ajuste; fallar con un mensaje explícito ante entradas incompatibles.
- [Los callbacks necesitan una señal de validación válida] → Rechazar la ejecución sin datos de validación, en vez de continuar con resultados no supervisados.
