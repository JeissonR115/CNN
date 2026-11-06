# Design

## Context

El repositorio contiene el plan y propuestas OpenSpec, pero todavía no tiene módulos de aplicación ni notebook. Los cambios relacionados fijan el preprocesamiento, la división, aumentación, CNN, evaluación y artefactos. Ver `proposal.md` y `specs/` para sus contratos.

## Goals / Non-Goals

**Goals:**
- Ofrecer una única implementación de entrenamiento consumible tanto desde la línea de comandos como desde el notebook versionado.
- Reutilizar exactamente los artefactos y preprocesamiento de entrenamiento en la API de inferencia que llamará Streamlit.
- Mantener el notebook legible, ejecutable desde un checkout y razonablemente pequeño en Git.

**Non-Goals:**
- Añadir una API HTTP, despliegue o entrenamiento dentro de Streamlit.
- Guardar datasets, credenciales, modelos entrenados ni salidas voluminosas del notebook en Git.

## Decisions

- Implementar `train(dataset_root)` en `src/train.py` como la única orquestación del flujo; aceptar `--data-dir` en la CLI y ejecutarla desde la raíz con `python -m src.train --data-dir <ruta>`. El flujo reutiliza los módulos de datos y modelo especificados en los cambios relacionados, verifica el guardado y devuelve el mejor modelo, el historial de ajuste y el orden de nombres de clase para permitir evaluación sin duplicar la lógica de entrenamiento. Se prefiere una función invocable y un CLI fino a mantener código de entrenamiento separado en el notebook.
- Mantener `notebooks/entrenamiento.ipynb` en Git como recorrido completo: exploración, llamada al entrenamiento común, evaluación final y ejemplo de `predict`. Las celdas deben usar rutas relativas a la raíz del checkout; documentar claramente la preparación del dataset. Limpiar salidas de celdas antes de guardar el notebook para no versionar imágenes, métricas o datos derivados innecesariamente.
- Implementar `predict(image)` en `src/predict.py` para aceptar exclusivamente imágenes PIL legibles, convertir a RGB, redimensionar a 128 × 128 y normalizar a `[0, 1]`. Cargar `models/cnn_classifier.keras` y `models/class_names.json` usando rutas ancladas a la raíz del repositorio, no al directorio de trabajo del proceso. Cargar el modelo una vez por proceso para evitar releerlo en cada solicitud de Streamlit y validar la dimensión softmax contra el listado de clases antes de devolver `(class_name, probability)`.
- Mantener los fallos de lectura, forma o incompatibilidad como errores explícitos; no ofrecer una clase predeterminada. El notebook local y Colab debe ejecutar desde el checkout/repositorio para que los imports y las rutas de artefactos sean los mismos.

## Risks / Trade-offs

- [El notebook puede desviarse de los módulos del proyecto] → Hacer que invoque `src.train` y `src.predict` y verificar que no replique su lógica de entrenamiento o inferencia.
- [Las rutas pueden variar entre Colab y ejecución local] → Resolver artefactos desde la ubicación del módulo y usar ruta de dataset explícita en la CLI y en la configuración documentada del notebook.
- [El caché de inferencia puede conservar un modelo anterior después de reemplazar artefactos en un proceso activo] → Documentar que se reinicie la aplicación después de reentrenar y reemplazar el modelo.
- [Las salidas del notebook pueden inflar el repositorio] → Guardar el notebook con salidas limpiadas y no incluir dataset ni modelo generado.
