# Tasks

## 1. Transformación de imágenes

- [x] 1.1 Implementar en un módulo compartido la conversión a RGB, el redimensionamiento a 128 × 128 y la normalización `float32` por 1/255; verificar con pruebas que la forma y el rango resultantes sean correctos para imágenes de distintos tamaños y modos.
- [ ] 1.2 Documentar en las instrucciones del flujo de entrenamiento la transformación y sus valores de entrada/salida; verificar que la documentación coincide con las pruebas del módulo compartido.

## 2. División estratificada

- [x] 2.1 Implementar la división 70/15/15 de rutas y etiquetas con estratificación y semilla 42, excluyendo archivos sin etiquetas; verificar proporciones con redondeo, representación por clase, repetibilidad y ausencia de solapamiento mediante pruebas.
- [ ] 2.2 Manejar clases con muestras insuficientes con un error explícito y añadir su prueba; documentar en el notebook que la división usa solo imágenes legibles etiquetadas y que los archivos del test oficial sin etiquetas se excluyen.

## 3. Integración de entrenamiento e inferencia

- [ ] 3.1 Conectar las particiones con el flujo de entrenamiento y reutilizar la transformación compartida en la predicción de Streamlit; verificar con una imagen de prueba que ambas rutas generan entradas con la misma forma, tipo, orden de canales y rango.
- [ ] 3.2 Ejecutar el flujo completo con un conjunto etiquetado pequeño de prueba; verificar que genera train/val/test disjuntos, conserva clases en cada partición cuando hay muestras suficientes y mantiene 70/15/15 dentro del redondeo.
