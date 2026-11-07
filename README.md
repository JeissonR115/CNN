# Clasificación de CIFAR-10 con CNN y Streamlit

Proyecto académico para entrenar una CNN con CIFAR-10 y probarla mediante una interfaz Streamlit.

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

El proyecto lee directamente el dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), sin convertir sus lotes a archivos de imagen. El archivo comprimido incluye cinco lotes de entrenamiento, un lote oficial de prueba, metadatos de clases y 60 000 imágenes RGB de 32 × 32 en diez clases.

El dataset ya extraído debe tener esta estructura:

```text
data/cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

Si todavía no lo descargaste, guarda el archivo `cifar-10-python.tar.gz` dentro de `data/` y extráelo allí. `.gitignore` excluye `data/`, así que el dataset y el archivo comprimido no se subirán a Git.

Las clases son `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship` y `truck`. El flujo usa los cinco lotes `data_batch_*` para crear una división estratificada 85 % train y 15 % validation; mantiene `test_batch` como prueba oficial y separado. No conviertas ni mezcles `test_batch` con los datos de entrenamiento. Las imágenes se convierten a RGB, se redimensionan a 128 × 128 y se normalizan a `float32` en `[0, 1]`.

La ruta predeterminada del notebook es `data/cifar-10-batches-py`. Si está en otro lugar, configura la ruta antes de iniciarlo:

   ```bash
   export CNN_DATASET_DIR="data/cifar-10-batches-py"
   ```

   En PowerShell:

   ```powershell
   $env:CNN_DATASET_DIR = "data\cifar-10-batches-py"
   ```

La exploración está en `notebooks/entrenamiento.ipynb`. Resume conteos y dimensiones y genera ejemplos reproducibles. También sigue siendo posible usar otro dataset organizado en carpetas etiquetadas.

## Entrenamiento

Desde la raíz del repositorio y con el entorno virtual activado, ejecuta el pipeline apuntando a la carpeta que contiene `batches.meta` y los lotes:

```bash
python -m src.train --data-dir "data/cifar-10-batches-py"
```

El entrenamiento usa los cinco lotes de entrenamiento; el test oficial se reserva para evaluación desde el notebook. Para otros datasets organizados en carpetas de clases, pasa como ruta la carpeta raíz que las contiene.

El entrenamiento genera `models/cnn_classifier.keras` (mejor checkpoint según `val_loss`) y `models/class_names.json` (arreglo de nombres en el mismo orden que las salidas softmax). Vuelve a cargar los artefactos y compara las predicciones antes de declarar el guardado como verificado. Ambos archivos se mantienen fuera de Git; entrena el modelo antes de iniciar la aplicación si no los tienes localmente.

Para una prueba breve del pipeline se puede limitar el número de épocas, sin superar el máximo de 30:

```bash
python -m src.train --data-dir "data/cifar-10-batches-py" --epochs 1
```

## Aplicación Streamlit

Con los dos artefactos generados por el entrenamiento:

```bash
streamlit run app.py
```

La interfaz acepta imágenes JPEG o PNG legibles y muestra la clase predicha junto con la probabilidad de esa predicción. El preprocesamiento y la inferencia se delegan a `src.predict.predict`; las imágenes ilegibles, artefactos faltantes o incompatibles producen un error visible y no una predicción por defecto. Si reemplazas los artefactos mientras Streamlit está activo, reinicia el proceso para que no conserve el modelo anterior.

## Notebook

`notebooks/entrenamiento.ipynb` recorre la exploración, división, vista previa de aumentación, entrenamiento, curvas, métricas del test oficial CIFAR-10, matriz de confusión y ejemplos de predicción. Usa `src.train` y `src.predict` en lugar de duplicar el pipeline. Ábrelo desde VS Code con el kernel del entorno virtual o desde Colab después de clonar el repositorio e instalar `requirements.txt`. Para datos fuera de la ruta predeterminada, configura `CNN_DATASET_DIR`; para una ejecución breve, `CNN_TRAINING_EPOCHS` acepta de 1 a 30 y su valor predeterminado es 30. Mantén las salidas del notebook limpias y no guardes imágenes del dataset, credenciales ni modelos generados.

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

En un checkout limpio solo se necesitan el código, `requirements.txt` y `README.md`: descarga y extrae CIFAR-10 localmente, y entrena para generar los artefactos antes de usar Streamlit. No se deben agregar datos o modelos a Git. Consulta [PLAN_INGENIERIA.md](PLAN_INGENIERIA.md) para el alcance y los criterios de aceptación.
