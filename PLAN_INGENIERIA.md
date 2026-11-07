# Plan de ingeniería: CNN + Streamlit

## 1. Objetivo

Desarrollar una aplicación que use una red neuronal convolucional (CNN) para clasificar imágenes en varias categorías. El proyecto incluirá el flujo de preparación de datos, entrenamiento y evaluación del modelo, y una interfaz en Streamlit que permita cargar una imagen y mostrar la clase predicha junto con su nivel de confianza.

El entrenamiento deberá poder ejecutarse en Google Colab, y la aplicación deberá poder iniciarse localmente siguiendo las instrucciones del repositorio.

## 2. Alcance

### Incluye

- Selección y documentación de un dataset público con entre 3 y 10 clases.
- Exploración de datos y preprocesamiento de imágenes.
- Separación de datos para entrenamiento, validación y prueba, evitando fuga de información.
- Implementación y entrenamiento de una CNN con TensorFlow/Keras.
- Evaluación con métricas apropiadas y visualizaciones.
- Aplicación Streamlit para predecir la clase de una imagen cargada.
- Instrucciones de instalación, entrenamiento y ejecución.

### No incluye

- Despliegue en producción ni disponibilidad permanente en la nube.
- Reentrenamiento del modelo desde la interfaz.
- Garantía de predicciones correctas para imágenes ajenas a las categorías del dataset.
- Inclusión del dataset completo o de artefactos grandes en el repositorio.

## 3. Dataset

**CIFAR-10**, distribuido por la Universidad de Toronto:
[https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- **Clases:** 10 — airplane, automobile, bird, cat, deer, dog, frog, horse, ship y truck.
- **Formato:** imágenes RGB de 32 × 32 almacenadas en lotes serializados de Python.
- **Tamaño:** 60 000 imágenes; 50 000 para entrenamiento y 10 000 en el lote oficial de prueba.
- **Uso previsto:** separar los cinco lotes de entrenamiento en train y validation; reservar `test_batch` para la evaluación final.

Descargar el archivo y extraerlo en `data/`. El código lee directamente los lotes y metadatos, sin convertir las imágenes a archivos individuales. No guardar datos voluminosos en Git.

## 4. Stack tecnológico

- **Python 3**
- **TensorFlow/Keras:** carga de datos, definición, entrenamiento y guardado del modelo.
- **Streamlit:** interfaz de carga y predicción.
- **NumPy y Pillow:** procesamiento de imágenes.
- **Matplotlib y scikit-learn:** métricas y visualizaciones.
- **Google Colab:** entorno recomendado para entrenamiento con GPU cuando esté disponible.
- **Git:** control de versiones del código y la documentación.

Las dimensiones, normalización y orden de clases usados al entrenar deben coincidir con el preprocesamiento de la aplicación.

## 5. Estructura propuesta del repositorio

```text
.
├── app.py                       # Interfaz Streamlit e inferencia
├── requirements.txt             # Dependencias de ejecución
├── README.md                    # Instalación, entrenamiento y uso
├── PLAN_INGENIERIA.md           # Este plan
├── notebooks/
│   └── entrenamiento.ipynb      # Exploración, entrenamiento y evaluación en Colab
├── src/
│   ├── data.py                  # Carga y preprocesamiento de datos
│   ├── model.py                 # Definición de la CNN
│   └── predict.py               # Preprocesamiento e inferencia compartidos
├── models/
│   └── .gitkeep                 # Ubicación prevista para el modelo entrenado
└── .gitignore                   # Excluye datos, credenciales y artefactos grandes
```

Los nombres pueden ajustarse a las convenciones del curso. No se deben versionar credenciales, datasets completos, cachés ni artefactos grandes sin autorización.

## 6. Flujo de trabajo

1. Descargar y explorar el dataset; documentar número de imágenes por clase y particiones.
2. Definir el tamaño de entrada, normalización y estrategia de aumento de datos.
3. Separar entrenamiento y validación de forma reproducible.
4. Entrenar una CNN base y guardar el historial de métricas.
5. Evaluar una sola vez con el conjunto de prueba y guardar métricas y matriz de confusión.
6. Exportar el modelo y los nombres de las clases para inferencia.
7. Implementar la carga de imágenes en Streamlit con el mismo preprocesamiento del entrenamiento.
8. Probar el flujo en Colab y la aplicación local; documentar los comandos.

## 7. Riesgos y mitigaciones

| Riesgo                                                              | Mitigación                                                                                                                                               |
| :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset incompleto o en una ruta distinta**                       | Documentar la estructura esperada, comprobar los lotes al inicio y permitir configurar la ruta del dataset.                                              |
| **Desbalance o variación entre clases**                             | Inspeccionar la distribución; reportar métricas por clase y considerar pesos de clase o aumento de datos.                                                |
| **Sobreajuste**                                                     | Usar validación, aumento de datos y regularización; comparar curvas de entrenamiento y validación.                                                       |
| **Fuga entre entrenamiento y prueba**                               | Mantener la partición de prueba separada hasta la evaluación final.                                                                                      |
| **Colab no dispone de GPU o tiene límites de sesión**               | Mantener un modelo y tamaño de entrada razonables; guardar checkpoints y documentar el tiempo de ejecución observado.                                    |
| **Diferencia de preprocesamiento entre entrenamiento e inferencia** | Compartir la misma función de preprocesamiento o verificar explícitamente dimensiones y normalización.                                                   |
| **Modelo o dependencias incompatibles**                             | Fijar versiones probadas y verificar la carga del modelo en el entorno de ejecución.                                                                     |
| **Predicción engañosa para imágenes fuera de las clases**           | Mostrar probabilidades como estimaciones, indicar las clases admitidas y advertir que el modelo no identifica necesariamente imágenes fuera del dataset. |

## 8. Criterios de aceptación

_Criterios genéricos, pendientes de sustituirse o ponderarse cuando se disponga de la rúbrica oficial:_

- [ ] CIFAR-10 está identificado con enlace, clases, tamaño y estructura local esperada.
- [ ] El notebook o flujo de entrenamiento se ejecuta en Colab desde la carga de datos hasta la evaluación, con instrucciones reproducibles.
- [ ] El código prepara las imágenes de forma consistente y conserva una partición de prueba independiente.
- [ ] El modelo CNN se entrena y se guarda en un formato que la aplicación puede cargar.
- [ ] Se reportan al menos exactitud y métricas por clase; se incluye una matriz de confusión o visualización equivalente.
- [ ] La aplicación Streamlit acepta una imagen compatible, valida la entrada y presenta una predicción legible con nivel de confianza.
- [ ] Una imagen de prueba puede recorrer el flujo completo sin error: carga, preprocesamiento, inferencia y presentación.
- [ ] `requirements.txt` y `README.md` documentan las dependencias, configuración necesaria y comandos para entrenar y ejecutar la aplicación.
- [ ] El repositorio no contiene credenciales ni datos voluminosos no autorizados.
- [ ] Las limitaciones conocidas y los riesgos relevantes están documentados.
