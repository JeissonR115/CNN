# Design

## Context

El repositorio contiene planificación, no una aplicación implementada. `scripts-train-predict` define `src.predict.predict(image)` como API compartida: recibe una imagen PIL y realiza conversión RGB, redimensionamiento a 128 × 128, normalización, inferencia y mapeo de clase usando los artefactos guardados. La interfaz debe pasarle la imagen y no reproducir esos pasos.

## Goals / Non-Goals

**Goals:**
- Mantener una sola ruta de inferencia y mostrar el resultado de forma clara.
- Evitar inicializar repetidamente el recurso de inferencia durante las reruns de Streamlit.
- Comunicar errores de lectura de imagen o de disponibilidad/compatibilidad del modelo.

**Non-Goals:**
- Mostrar la distribución completa de probabilidades por clase; la barra representa solo la confianza de la clase predicha.
- Entrenar o reemplazar el modelo desde la aplicación.
- Añadir servicios de despliegue o persistencia de imágenes cargadas.

## Decisions

- Implementar `app.py` como una interfaz delgada: restringir el cargador a JPEG/PNG, abrir el archivo como imagen PIL y pasarlo sin transformación adicional a `src.predict.predict(image)`. Así, el preprocesamiento se mantiene en el predictor y coincide con el entrenamiento.
- Usar `@st.cache_resource` en un getter de la API compartida para conservar el mismo recurso callable entre reruns. El predictor sigue siendo responsable de su modelo y de cargar sus artefactos una sola vez; la aplicación no debe cargar una segunda copia del modelo ni crear otra ruta de inferencia.
- Presentar la clase devuelta y su confianza como porcentaje y barra visual cuyo valor se mantiene en `[0, 1]`. La aplicación solo muestra resultados después de una predicción válida.
- Informar fallos esperados de decodificación e inferencia con un mensaje visible mediante Streamlit, sin asignar una clase por defecto ni silenciar errores. La carga directa del modelo desde `app.py` se descarta porque duplicaría la lógica del predictor.
- Tras sustituir los artefactos en disco, reiniciar el proceso de Streamlit para evitar que el caché siga usando el recurso anterior.

## Risks / Trade-offs

- [Los artefactos no existen antes del entrenamiento] → Mostrar un error de disponibilidad que indique que se deben generar los artefactos requeridos antes de predecir.
- [Un archivo puede tener extensión admitida pero contenido inválido] → Validar al abrir/decodificar la imagen y no llamar al predictor si la lectura falla.
- [El caché puede conservar el modelo anterior tras reentrenar] → Documentar el reinicio de la aplicación después de reemplazar los artefactos.
