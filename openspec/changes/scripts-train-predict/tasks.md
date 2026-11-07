# Tasks

## 1. Punto de entrada de entrenamiento

- [x] 1.1 Implementar `train(dataset_root)` en `src/train.py` y una CLI `python -m src.train --data-dir ...` que use el flujo compartido de datos/modelo, rechace rutas inválidas y guarde/verifique los artefactos; verificar la CLI y la función con un dataset pequeño de prueba.
- [x] 1.2 Devolver desde la función el mejor modelo, historial y nombres de clase para evaluación posterior; verificar que la evaluación puede usar esos resultados sin repetir la lógica de entrenamiento.
- [x] 1.3 Documentar la preparación del dataset y el comando de entrenamiento; verificar desde la raíz del repo que el comando se ejecuta y errores de dataset se presentan explícitamente.

## 2. API de predicción

- [x] 2.1 Implementar `predict(image)` en `src/predict.py` para aceptar PIL, aplicar RGB/128 × 128/normalización `[0, 1]`, cargar los artefactos desde rutas estables y devolver clase y probabilidad máxima; verificar con imágenes RGB y no RGB, tamaño distinto y un modelo de prueba conocido.
- [x] 2.2 Validar entradas, carga de modelo y correspondencia de `class_names.json`, sin retornos por defecto; verificar con entrada inválida, artefactos ausentes, JSON malformado y dimensión incompatible.
- [x] 2.3 Integrar la API con la aplicación Streamlit y documentar que el proceso debe reiniciarse tras reemplazar artefactos; verificar que una imagen cargada produce etiqueta/confianza y que fallos se muestran como error.

## 3. Notebook versionado e integración

- [x] 3.1 Crear `notebooks/entrenamiento.ipynb` con el flujo de exploración, entrenamiento, evaluación y predicción usando `src.train` y `src.predict`; verificar que el notebook es JSON válido, se ejecuta en orden desde el checkout con dataset configurado y no duplica el pipeline.
- [x] 3.2 Guardar el notebook con salidas limpiadas y sin dataset, credenciales ni modelos generados; verificar el diff del archivo y las reglas de exclusión del repositorio.
- [x] 3.3 Ejecutar una verificación de integración del flujo notebook/CLI/inferencia con un dataset pequeño; confirmar que se conservan orden de clases, rutas de artefactos y preprocesamiento de entrenamiento a predicción.
