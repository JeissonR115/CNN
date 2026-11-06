# Design

## Context

El repositorio aún no contiene notebook de entrenamiento ni evaluación. El plan prevé evaluación final en test con métricas y matriz de confusión, y `modelo-cnn` define un checkpoint elegido únicamente por `val_loss`. Ver `proposal.md` para la motivación y `specs/model-evaluation/spec.md` para el contrato.

## Goals / Non-Goals

**Goals:**
- Producir una evaluación final legible y repetible a partir del modelo seleccionado, el historial de entrenamiento y las etiquetas de test.
- Mantener una correspondencia única entre índices de clase y nombres en reportes, matrices y predicciones.

**Non-Goals:**
- Crear una interfaz de evaluación interactiva o cambiar el modelo según los resultados del test.
- Añadir un umbral de accuracy mínimo para aprobar el clasificador.

## Decisions

- Ejecutar la evaluación tras finalizar el entrenamiento y cargar/restaurar el mejor checkpoint seleccionado con validación. El test no se usa en callbacks ni para escoger el modelo; se evitan evaluaciones repetidas durante el ajuste para preservar su papel como conjunto final independiente.
- Calcular métricas sobre el conjunto test completo. Usar un orden explícito de clases para `classification_report` y matriz de confusión, de modo que las clases sin predicciones también aparezcan y que filas y columnas sean comparables.
- Graficar las series `loss`/`val_loss` y `accuracy`/`val_accuracy` guardadas por el ajuste, con épocas en el eje x. Si falta alguna, informar el nombre de la serie ausente en vez de presentar una gráfica incompleta como válida.
- Seleccionar hasta 10 índices de test sin reemplazo con semilla fija 42 para la tabla de ejemplos. Para cada uno mostrar la probabilidad asociada al índice de la clase predicha, no la probabilidad de la clase real. Si hay menos de 10 muestras, mostrar todas y señalar la limitación.

## Risks / Trade-offs

- [Un orden de clases inconsistente puede invalidar métricas y etiquetas] → Pasar la misma lista de nombres al reporte, la matriz y la conversión de índices a clases.
- [Una muestra pequeña de predicciones puede ocultar errores de otras clases] → Calcular métricas agregadas y matriz sobre todo el test; usar las 10 predicciones como ejemplos explicativos, no como sustituto.
- [La selección aleatoria podría repetir muestras entre ejecuciones] → Fijar explícitamente la semilla para la muestra y seleccionar sin reemplazo.
