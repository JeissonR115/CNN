# Clasificación de imágenes con CNN y Streamlit

Proyecto académico para entrenar una CNN con Intel Image Classification y probarla mediante una interfaz Streamlit.

## Requisitos

- Python 3.10 o posterior.
- TensorFlow con una rueda disponible para la versión de Python y el sistema operativo elegidos.
- Git para clonar el proyecto.

La resolución e importación de las dependencias de `requirements.txt` se verificaron en Linux con Python 3.12.3. La disponibilidad de TensorFlow puede variar según la plataforma y la versión de Python.

## Instalación

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

El dataset candidato es [Intel Image Classification en Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). La página de Kaggle describe aproximadamente 25 000 imágenes de 150 × 150 en seis clases: `buildings`, `forest`, `glacier`, `mountain`, `sea` y `street`; son cifras publicadas por la fuente, no mediciones de los archivos locales.

1. Descarga y extrae el dataset fuera del repositorio, después de revisar sus condiciones de acceso y licencia.
2. No agregues al repositorio las imágenes, archivos comprimidos ni credenciales de Kaggle.
3. Usa como ruta de datos la carpeta extraída que contiene las imágenes etiquetadas organizadas por clase. El flujo de entrenamiento valida la estructura disponible; las particiones sin etiquetas no se usan para la división estratificada.
4. Configura la ruta antes de iniciar el notebook:

   ```bash
   export CNN_DATASET_DIR="/ruta/al/dataset"
   ```

   En Colab, asigna la ruta extraída a `CNN_DATASET_DIR` en una celda antes de ejecutar la exploración.

La exploración está en `notebooks/entrenamiento.ipynb`. Resume conteos y dimensiones observadas, identifica archivos ilegibles y genera una cuadrícula reproducible; si el dataset no está disponible o no tiene entre 3 y 10 clases legibles, informa el bloqueo en lugar de validarlo.

## Entrenamiento

Desde la raíz del repositorio y con el entorno virtual activado, ejecuta el pipeline con la ruta local del dataset:

```bash
python -m src.train --data-dir "/ruta/al/dataset"
```

El entrenamiento genera `models/cnn_classifier.keras` y `models/class_names.json`. Esos artefactos se mantienen fuera de Git; debes entrenar el modelo antes de iniciar la aplicación si no los tienes localmente.

> El comando estará disponible al implementar `src/train.py` según el contrato del proyecto.

## Aplicación Streamlit

Con los artefactos generados por el entrenamiento:

```bash
streamlit run app.py
```

La interfaz acepta imágenes JPEG o PNG y muestra la clase predicha junto con la confianza de esa predicción. El preprocesamiento y la inferencia se delegan al predictor compartido. Si reemplazas los artefactos mientras Streamlit está activo, reinicia el proceso para que no conserve el modelo anterior.

> La aplicación estará disponible al implementar `app.py` e `src/predict.py` según los contratos del proyecto.

## Notebook

El recorrido de exploración, entrenamiento, evaluación y predicción estará en `notebooks/entrenamiento.ipynb`. Se puede abrir desde VS Code o usar en Colab después de clonar el repositorio e instalar `requirements.txt`. Configura allí la ruta al dataset y no guardes salidas con imágenes, credenciales ni artefactos generados.

## Estructura final prevista

```text
.
├── app.py
├── requirements.txt
├── README.md
├── PLAN_INGENIERIA.md
├── .gitignore
├── notebooks/
│   └── entrenamiento.ipynb
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
└── models/
    ├── .gitkeep
    ├── cnn_classifier.keras       # Generado localmente; ignorado por Git
    └── class_names.json           # Generado localmente; ignorado por Git
```

Consulta [PLAN_INGENIERIA.md](PLAN_INGENIERIA.md) para el alcance, el flujo de trabajo y los criterios de aceptación.
