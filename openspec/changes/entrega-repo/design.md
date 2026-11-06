# Design

## Context

El repositorio contiene actualmente un README mínimo y un plan de ingeniería, pero no manifiesto de dependencias ni reglas de exclusión. Las propuestas relacionadas definen `python -m src.train --data-dir <ruta>`, `streamlit run app.py`, un notebook en `notebooks/entrenamiento.ipynb`, y artefactos generados bajo `models/`.

## Goals / Non-Goals

**Goals:**
- Que una persona pueda seguir el flujo local desde un checkout limpio con los prerequisitos y artefactos correctamente identificados.
- Mantener una única fuente de verdad de instrucciones para CLI, aplicación, notebook y estructura del repositorio.
- Evitar que archivos locales sensibles y grandes se incorporen accidentalmente a Git.

**Non-Goals:**
- Garantizar compatibilidad con toda versión futura de Python o TensorFlow.
- Distribuir las credenciales de Kaggle, el dataset ni los modelos entrenados junto con el código.
- Añadir instalación como paquete Python ni publicar el proyecto en un índice de paquetes.

## Decisions

- Declarar Python 3.10 como versión mínima en `requirements.txt`/README y documentar las versiones efectivamente compatibles con las dependencias seleccionadas. “3.10+” expresa una versión mínima; TensorFlow y otras dependencias pueden limitar versiones mayores concretas. Se evita prometer soporte para intérpretes donde no existan wheels compatibles.
- Mantener `requirements.txt` como manifiesto de instalación único para los flujos de entrenamiento y aplicación, con las dependencias directas necesarias, incluyendo Streamlit, TensorFlow, NumPy, Pillow, Matplotlib y scikit-learn. Las versiones deben ser coherentes entre sí y con Python admitido; no fijar versiones hipotéticas antes de validar el entorno objetivo.
- Organizar `README.md` como guía operativa: prerequisitos/entorno, instalación, obtención y ubicación del dataset Intel Image Classification, entrenamiento con `python -m src.train --data-dir <ruta>`, creación de artefactos, ejecución con `streamlit run app.py`, y uso del notebook. Incluir la estructura final con `app.py`, `src/`, `notebooks/`, `models/`, `requirements.txt`, `README.md`, `PLAN_INGENIERIA.md` y `.gitignore`.
- Configurar `.gitignore` para patrones acotados de entornos virtuales, caches, datasets locales, credenciales como `kaggle.json`, salidas generadas de notebooks y artefactos del modelo; conservar archivos de código y directorios necesarios mediante `.gitkeep` cuando corresponda. Se excluyen pesos generados porque el flujo documentado los crea antes de inferencia.
- Evitar duplicar en README instrucciones independientes que contradigan las propuestas existentes. Si el nombre o los argumentos de los comandos cambian durante implementación, actualizar en conjunto la documentación y los tests/documentos relacionados.

## Risks / Trade-offs

- [TensorFlow no ofrece ruedas para toda versión futura de Python] → Declarar el mínimo solicitado y enumerar las versiones verificadas/compatibles en vez de prometer soporte indefinido.
- [Un `.gitignore` demasiado amplio puede ocultar archivos necesarios] → Usar patrones específicos y verificar casos representativos con `git check-ignore`; revisar los archivos de código/documentación que sí deben quedar versionados.
- [La guía puede desviarse de los comandos implementados por cambios relacionados] → Tratar los contratos de `scripts-train-predict` y `app-streamlit` como fuente de verdad y validar los comandos del README al implementar.
