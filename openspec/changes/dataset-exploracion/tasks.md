# Tasks

## 1. Selección y metadatos del dataset

- [ ] 1.1 Añadir al notebook de entrenamiento la identificación de la fuente Kaggle, lectura de clases y particiones y validación del rango de 3 a 10 clases; verificar con el candidato Intel y con casos de dataset inaccesible o fuera del rango.
- [ ] 1.2 Documentar el acceso al dataset y los resultados observados, diferenciando los datos medidos de las descripciones publicadas; verificar que el flujo no expone credenciales y que el informe no valida un dataset inaccesible.

## 2. Distribución y dimensiones

- [ ] 2.1 Generar conteos exactos de imágenes legibles por clase y partición, con totales reconciliables; verificar sumas con un conjunto pequeño de datos de prueba que incluya una clase vacía y archivos ilegibles.
- [ ] 2.2 Medir y resumir cada par de dimensiones observado y su cantidad; verificar el resultado con imágenes de tamaño uniforme y con tamaños distintos.

## 3. Cuadrícula de ejemplos e integración

- [ ] 3.1 Generar una cuadrícula con al menos un ejemplo legible y etiqueta correcta por clase, usando selección reproducible; verificar cobertura de clases y el aviso cuando falte un ejemplo legible.
- [ ] 3.2 Ejecutar el flujo de exploración de principio a fin con el dataset descargado y revisar que fuente, clases, distribución, dimensiones y cuadrícula estén disponibles antes del entrenamiento.
- [ ] 3.3 Actualizar las instrucciones del proyecto para indicar cómo obtener los datos y ejecutar la exploración; verificar que las instrucciones coincidan con el flujo implementado.
