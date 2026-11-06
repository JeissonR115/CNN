# Design

## Context

El proyecto aún no contiene notebook ni pipeline de entrenamiento. El plan prevé TensorFlow/Keras y Matplotlib en `notebooks/entrenamiento.ipynb`; la división de datos se especifica por separado en `preprocesamiento-division`. Ver `proposal.md` para el motivo y `specs/data-augmentation/spec.md` para el contrato.

## Goals / Non-Goals

**Goals:**
- Integrar transformaciones aleatorias solo en el pipeline de entrenamiento y poder inspeccionar sus efectos antes del ajuste del modelo.
- Mantener el preprocesamiento base de 128 × 128 RGB y rango `[0, 1]` establecido para los datos.

**Non-Goals:**
- Modificar las imágenes almacenadas o guardar copias aumentadas como dataset.
- Aumentar validation o test, ni cambiar el procedimiento de partición.
- Añadir aumentaciones geométricas o de color distintas a las solicitadas.

## Decisions

- Aplicar las transformaciones en línea al lote de entrenamiento después de crear las particiones, mediante la lógica de datos propuesta en `src/data.py`. Esto evita duplicar datos en disco y previene contaminación de validation/test. La alternativa de materializar versiones aumentadas se descarta porque consume almacenamiento y dificulta mantener claro qué partición originó cada imagen.
- Usar transformaciones aleatorias equivalentes a: flip horizontal con probabilidad 0.5; rotación uniforme en `[-15°, 15°]`; zoom uniforme en `[0.9, 1.1]`; y factor de contraste uniforme en `[0.8, 1.2]`. No usar flip vertical. Configurar el contraste para entradas en `[0, 1]`, limitar los valores de salida a ese rango y usar relleno reflectante en las transformaciones geométricas para evitar bordes negros.
- Conservar el pipeline de inferencia y evaluación sin aumentación; las capas de aumento se conectan únicamente al flujo de entrenamiento, no al modelo de predicción usado por Streamlit.
- Generar la vista previa a partir de una muestra reproducible del conjunto de entrenamiento: al menos una imagen por clase, la original, cada efecto individual y la combinación. La selección y las transformaciones de la vista previa usan una semilla fija para que las comparaciones puedan repetirse. Matplotlib ya está contemplado en el plan, por lo que no se incorpora una dependencia nueva.

## Risks / Trade-offs

- [La rotación y el zoom pueden introducir píxeles de borde] → Usar un modo de relleno no constante adecuado para imágenes RGB y revisar esos bordes en la cuadrícula.
- [El contraste o la interpolación pueden producir valores fuera de la escala esperada] → Limitar y comprobar que la salida permanece finita y en `[0, 1]`.
- [Una vista previa determinista no refleja toda la variedad aleatoria del entrenamiento] → Usarla como inspección de cada tipo de transformación, no como una estimación de todas las muestras posibles.
