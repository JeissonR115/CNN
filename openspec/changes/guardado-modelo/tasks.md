# Tasks

## 1. Guardado de artefactos

- [x] 1.1 Configurar el checkpoint para guardar el mejor modelo por `val_loss` como `models/cnn_classifier.keras`; verificar tras un entrenamiento corto que el archivo existe y puede abrirse como modelo Keras.
- [x] 1.2 Guardar `models/class_names.json` como arreglo JSON ordenado exactamente según las etiquetas de salida y validar nombres únicos/no vacíos; verificar con los nombres de clase del cargador que índices y nombres coinciden.

## 2. Verificación de carga

- [x] 2.1 Añadir una comprobación que vuelva a cargar el modelo y el JSON, valide dimensión de salida y longitud del arreglo, y compare predicciones antes y después del guardado con tolerancias abs 1e-6 y rel 1e-5; verificar los casos correcto y desalineado.
- [x] 2.2 Mostrar un error explícito y no marcar el guardado como exitoso si falta un archivo, falla la carga, difieren dimensiones o las predicciones exceden la tolerancia; verificar con artefactos inválidos y metadatos incompatibles.

## 3. Integración de inferencia y documentación

- [x] 3.1 Cargar ambos artefactos desde Streamlit y validar al inicio la correspondencia entre cantidad de clases y salidas; verificar que una imagen produce una clase nombrada y que los archivos ausentes o incompatibles notifican un error visible.
- [x] 3.2 Documentar las rutas, el formato de `class_names.json` y la verificación de carga; verificar que las instrucciones coinciden con los nombres y el flujo implementados.
