# Design

## Context

El repositorio contiene el plan de ingeniería, pero todavía no tiene notebook de entrenamiento ni código de inferencia. El plan propone TensorFlow/Keras, Streamlit y scikit-learn, y prevé un módulo `src/` compartido entre el notebook y la aplicación. Ver `proposal.md` para el alcance y `specs/dataset-preprocessing/spec.md` para el contrato de comportamiento.

## Goals / Non-Goals

**Goals:**
- Aplicar una sola definición de transformación de imagen al entrenamiento y a la predicción.
- Producir particiones disjuntas y reproducibles sobre los datos etiquetados disponibles.

**Non-Goals:**
- Añadir aumentación de datos, balanceo de clases o transformaciones aprendidas a partir del dataset.
- Usar para evaluación imágenes de Kaggle que no tengan etiquetas.

## Decisions

- Centralizar la transformación RGB, el redimensionamiento a 128 × 128 y el escalado `float32` por 1/255 en un módulo compartido bajo `src/`; el notebook y Streamlit deben llamar la misma lógica, evitando implementaciones duplicadas.
- Dividir las rutas y etiquetas antes de cualquier aumentación de datos. Usar una división estratificada en dos etapas: separar el 30 % de reserva del 70 % de entrenamiento y dividir la reserva en dos mitades para validación y prueba. Fijar `random_state=42` en ambas etapas para que el resultado sea repetible. Se elige este enfoque por usar scikit-learn, ya contemplado en el plan, y mantener el flujo sencillo.
- Aceptar redondeo de cantidades por clase y entre particiones, ya que las imágenes son indivisibles. Si una clase no tiene suficientes ejemplos para aparecer en las tres particiones, detener la división con un error claro en vez de eliminar la estratificación.
- Construir las tres particiones solo a partir de archivos legibles con etiquetas. El directorio oficial sin etiquetas se excluye porque no permite evaluar resultados.

## Risks / Trade-offs

- [Las cantidades pequeñas por clase pueden impedir una división estratificada o alejar los porcentajes exactos] → Validar los conteos antes de dividir, explicar la clase insuficiente y no usar silenciosamente una estrategia alternativa.
- [El notebook y la aplicación podrían desviarse si replican el preprocesamiento] → Mantener una implementación compartida y verificar que ambos consumidores produzcan tensores con la misma forma, tipo y rango.
- [La semilla fija facilita reproducir experimentos, pero no mide variación entre particiones] → Conservar la semilla como constante explícita; evaluar distintas semillas queda fuera de este cambio.
