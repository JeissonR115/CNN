# Tasks

## 1. Arquitectura del clasificador

- [x] 1.1 Implementar en `src/model.py` el constructor CNN con tres bloques de dos Conv2D (32/64/128 filtros), BatchNormalization, MaxPooling, Dropout 0.25, Dense 128, Dropout 0.5 y softmax configurable por clases; verificar con pruebas la forma de entrada 128 × 128 × 3, cantidad de salidas, suma de probabilidades y capas requeridas para 6 y para otro número válido de clases.
- [ ] 1.2 Documentar la arquitectura y sus formas en el notebook de entrenamiento; verificar que `model.summary()` aparece antes de iniciar `fit` con formas y conteos de parámetros.

## 2. Configuración de entrenamiento y callbacks

- [x] 2.1 Implementar compilación y ajuste con Adam 0.001, sparse categorical crossentropy, accuracy, batch 32 y máximo 30 épocas; verificar con una prueba de configuración o ajuste corto que los parámetros coinciden y que se rechaza la ausencia de validación.
- [x] 2.2 Configurar EarlyStopping (val_loss, paciencia 5, restaurar mejores pesos), ModelCheckpoint (guardar solo mínimo val_loss) y ReduceLROnPlateau (factor 0.5, paciencia 2, mínimo 1e-6); verificar reducción de LR, detención, restauración y selección del mejor checkpoint con datos de validación controlados.
- [ ] 2.3 Documentar en el notebook el criterio de checkpoint y que test no participa en callbacks; verificar que el conjunto de prueba se evalúa únicamente después de seleccionar/restaurar el mejor modelo.

## 3. Integración del modelo

- [ ] 3.1 Ejecutar una prueba de integración con entradas 128 × 128 RGB normalizadas y etiquetas enteras; verificar que el modelo produce una salida por clase, entrena con validación y guarda un checkpoint cargable para la etapa de inferencia.
