# Proposal

## Why

El proyecto necesita una descripción reproducible del dataset antes de entrenar la CNN: conocer su procedencia, clases, balance y características de las imágenes permite detectar problemas de calidad y justificar su adecuación. La exploración también debe producir evidencia visual revisable, en vez de depender de cifras aproximadas o inspección manual.

## What Changes

- Incorporar una exploración reproducible del candidato Intel Image Classification de Kaggle, confirmando su estructura, las clases presentes y las particiones disponibles.
- Generar un resumen de cantidad de imágenes por clase y dimensiones de imagen observadas, distinguiendo datos medidos de características publicadas por la fuente.
- Generar una cuadrícula de ejemplos etiquetados que permita inspeccionar visualmente las clases.
- Documentar la fuente, el procedimiento y los resultados de exploración, de forma que puedan revisarse antes del entrenamiento.

## Capabilities

### New Capabilities
- `dataset-exploracion`: selección y descripción verificable de un dataset de imágenes, incluyendo distribución de clases, tamaños de imagen y ejemplos visuales.

### Modified Capabilities
- Ninguna.

## Impact

- Afecta al flujo de exploración documentado para el notebook de entrenamiento previsto en `notebooks/entrenamiento.ipynb`; el repositorio actualmente no contiene ese notebook ni código de aplicación.
- Usa el dataset público Intel Image Classification de Kaggle como candidato inicial (6 clases; dimensiones descritas aproximadamente como 150 × 150 hasta confirmar los archivos descargados).
- No requiere cambios de dependencias ni servicios externos aparte del acceso al dataset. Las credenciales de Kaggle, si fueran necesarias, deben configurarse fuera del repositorio.
