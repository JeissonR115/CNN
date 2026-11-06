# Proposal

## Why

El proyecto necesita una interfaz sencilla para probar el clasificador con imágenes propias y comunicar el resultado de forma clara. La aplicación debe reutilizar la API común de predicción para evitar discrepancias con el preprocesamiento y el mapeo de clases del entrenamiento.

## What Changes

- Añadir `app.py` como punto de entrada de Streamlit, con carga de imagen, presentación de la clase predicha y una barra que represente la confianza de esa predicción.
- Usar `@st.cache_resource` para inicializar una sola vez por proceso el recurso compartido de inferencia, sin duplicar en la interfaz la carga/preprocesamiento/predicción que corresponde a `src.predict`.
- Mostrar errores claros cuando no se pueda leer una imagen o no estén disponibles los artefactos requeridos por el predictor.

## Capabilities

### New Capabilities
- `streamlit-app`: interacción de usuario para cargar una imagen y mostrar el resultado y la confianza del clasificador.

### Modified Capabilities
- Ninguna.

## Impact

- Añade `app.py` y enlaza Streamlit con `src.predict.predict(image)` y los artefactos `models/cnn_classifier.keras` y `models/class_names.json` mediante la infraestructura compartida de inferencia.
- Requiere Streamlit en el entorno de ejecución; el stack ya lo contempla en `PLAN_INGENIERIA.md`.
- La barra representa solo la probabilidad de la clase predicha, conforme al contrato actual `(clase, confianza)`; no muestra una distribución completa por clase.
