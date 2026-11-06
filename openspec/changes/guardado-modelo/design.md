# Design

## Context

El plan reserva `models/` para artefactos, pero el repositorio todavía no contiene el modelo, el notebook ni la aplicación Streamlit. El cambio `modelo-cnn` propone seleccionar el checkpoint mediante `val_loss`; `evaluacion` propone usarlo posteriormente y mantener test fuera de la selección. Ver `proposal.md` para la motivación y `specs/model-artifacts/spec.md` para el contrato.

## Goals / Non-Goals

**Goals:**
- Producir un modelo serializado y metadatos de clases que puedan cargarse como una unidad lógica en evaluación e inferencia.
- Detectar errores de serialización y desalineación de clases antes de marcar los artefactos como listos.

**Non-Goals:**
- Versionado remoto de modelos, registro de experimentos o gestión de múltiples versiones.
- Persistir dataset, pesos intermedios o credenciales.

## Decisions

- Guardar el modelo en `models/cnn_classifier.keras`. Configurar el `ModelCheckpoint` de `modelo-cnn` para escribir el mejor modelo según mínimo `val_loss` directamente en ese destino, con `save_best_only=True`; así se conserva el checkpoint seleccionado sin hacer un guardado final que pudiera sobrescribirlo con una época inferior.
- Escribir `models/class_names.json` como arreglo UTF-8 de cadenas, por ejemplo `["buildings", "forest", "glacier", "mountain", "sea", "street"]`, respetando exactamente el orden de índices que entrega el cargador etiquetado. Una lista plana coincide directamente con el índice de salida softmax y evita ambigüedades de mapas con orientación variable.
- Verificar tras finalizar el ajuste que ambos archivos existen, recargar el modelo sin objetos personalizados, parsear el JSON y comprobar nombres únicos/no vacíos y coincidencia entre su longitud y la dimensión de salida. Comparar predicciones en modo inferencia sobre una entrada de muestra antes y después de guardar, usando `numpy.testing.assert_allclose` con tolerancia absoluta 1e-6 y relativa 1e-5. El test no interviene en esta comprobación ni en la selección del checkpoint.
- Mantener en Streamlit una carga explícita de esos dos artefactos y validar al inicio que el número de nombres coincide con el número de salidas del modelo; errores de archivo o incompatibilidad deben ser visibles y no convertirse en etiquetas por defecto.

## Risks / Trade-offs

- [El orden de clases puede cambiar si se vuelve a crear el cargador] → Guardar el orden derivado del pipeline de etiquetas usado en entrenamiento y verificar su tamaño al recargar.
- [Un modelo serializado puede comportarse distinto al original] → Comprobar equivalencia numérica de las predicciones en inferencia antes de aceptar los archivos.
- [La carpeta `models/` puede no existir al ejecutar en un entorno limpio] → Crear la carpeta como parte del flujo de guardado y verificar los archivos resultantes.
