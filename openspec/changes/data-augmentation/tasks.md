# Tasks

## 1. Aumentación exclusiva de train

- [x] 1.1 Implementar en `src/data.py` las transformaciones aleatorias acordadas y conectarlas solo a train después de la división; verificar con pruebas que flip, rotación, zoom y contraste respeten sus rangos, que las salidas sean finitas y estén en `[0, 1]`, y que validation/test no cambien.
- [x] 1.2 Documentar en el notebook de entrenamiento el orden de división y aumentación; verificar que la vista de entrenamiento use el pipeline aumentado y que validación y prueba usen únicamente el preprocesamiento base.

## 2. Visualización de transformaciones

- [x] 2.1 Crear en el notebook una cuadrícula reproducible con al menos una imagen train por clase, mostrando original, flip, rotación, zoom, contraste y combinación, con etiquetas de clase y transformación; verificar que repeticiones con las mismas semillas muestren las mismas muestras y resultados.
- [x] 2.2 Añadir al notebook una descripción de los rangos y del uso de la cuadrícula para revisar artefactos; verificar que cada transformación mostrada corresponda a los parámetros especificados.

## 3. Integración del entrenamiento

- [x] 3.1 Conectar el pipeline aumentado al entrenamiento de la CNN sin aplicar aumentación en validación, test ni predicción Streamlit; verificar con un lote de cada partición que solo el de entrenamiento puede variar entre iteraciones y que las demás entradas se mantienen estables.
