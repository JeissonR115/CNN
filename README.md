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
3. Usa como ruta de datos la carpeta extraída que contiene las particiones y directorios de clase. El flujo incorpora solo imágenes etiquetadas legibles; los archivos sin etiquetas se excluyen.
4. El preprocesamiento compartido convierte cada imagen a RGB, la redimensiona a 128 × 128 y normaliza los canales `float32` a `[0, 1]` dividiendo los valores de 8 bits por 255. Las rutas etiquetadas se separan estratificadamente en 70 % train, 15 % validation y 15 % test con semilla 42. Las particiones son disjuntas; si no se puede conservar cada clase en las tres, el flujo falla explícitamente.
5. Configura la ruta antes de iniciar el notebook:

   ```bash
   export CNN_DATASET_DIR="/ruta/al/dataset"
   ```

   En PowerShell:

   ```powershell
   $env:CNN_DATASET_DIR = "C:\ruta\al\dataset"
   ```

La exploración está en `notebooks/entrenamiento.ipynb`. Resume conteos y dimensiones observadas, identifica archivos ilegibles y genera ejemplos reproducibles. Si el dataset no está disponible o no tiene entre 3 y 10 clases legibles, informa el bloqueo y no inicia entrenamiento.

## Entrenamiento

Desde la raíz del repositorio y con el entorno virtual activado, ejecuta el pipeline con la ruta local del dataset:

```bash
python -m src.train --data-dir "/ruta/al/dataset"
```

El entrenamiento genera `models/cnn_classifier.keras` (mejor checkpoint según `val_loss`) y `models/class_names.json` (arreglo de nombres en el mismo orden que las salidas softmax). Vuelve a cargar los artefactos y compara las predicciones antes de declarar el guardado como verificado. Ambos archivos se mantienen fuera de Git; entrena el modelo antes de iniciar la aplicación si no los tienes localmente.

Para una prueba breve del pipeline se puede limitar el número de épocas, sin superar el máximo de 30:

```bash
python -m src.train --data-dir "/ruta/al/dataset" --epochs 1
```

## Aplicación Streamlit

Con los dos artefactos generados por el entrenamiento:

```bash
streamlit run app.py
```

La interfaz acepta imágenes JPEG o PNG legibles y muestra la clase predicha junto con la probabilidad de esa predicción. El preprocesamiento y la inferencia se delegan a `src.predict.predict`; las imágenes ilegibles, artefactos faltantes o incompatibles producen un error visible y no una predicción por defecto. Si reemplazas los artefactos mientras Streamlit está activo, reinicia el proceso para que no conserve el modelo anterior.

## Notebook

`notebooks/entrenamiento.ipynb` recorre la exploración, división, vista previa de aumentación, entrenamiento, curvas, métricas de test, matriz de confusión y ejemplos de predicción. Usa `src.train` y `src.predict` en lugar de duplicar el pipeline. Ábrelo desde VS Code con el kernel del entorno virtual o desde Colab después de clonar el repositorio e instalar `requirements.txt`. Configura `CNN_DATASET_DIR`; para una ejecución breve, `CNN_TRAINING_EPOCHS` acepta de 1 a 30 y su valor predeterminado es 30. Mantén las salidas del notebook limpias y no guardes imágenes del dataset, credenciales ni modelos generados.

## Estructura del repositorio

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
│   ├── exploration.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── test_app.py
│   ├── test_data.py
│   ├── test_exploration.py
│   ├── test_model.py
│   ├── test_notebook.py
│   ├── test_predict.py
│   └── test_train.py
└── models/
    ├── .gitkeep
    ├── cnn_classifier.keras       # Generado localmente; ignorado por Git
    └── class_names.json           # Generado localmente; ignorado por Git
```

En un checkout limpio solo se necesitan el código, `requirements.txt` y `README.md`: descarga el dataset localmente y entrena para generar los artefactos antes de usar Streamlit. No se requieren credenciales de Kaggle ni se deben agregar datos o modelos a Git. Consulta [PLAN_INGENIERIA.md](PLAN_INGENIERIA.md) para el alcance y los criterios de aceptación.
