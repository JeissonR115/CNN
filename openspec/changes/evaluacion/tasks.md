# Tasks

## 1. Métricas finales

- [ ] 1.1 Implementar en el notebook la evaluación del mejor checkpoint con el test completo, reportando accuracy, classification_report y matriz de confusión con orden de clases explícito; verificar con etiquetas de prueba controladas que métricas, soportes y ejes de la matriz concuerdan.
- [ ] 1.2 Manejar test vacío con un mensaje explícito y sin producir métricas ficticias; verificar este comportamiento con un conjunto vacío.

## 2. Curvas de entrenamiento

- [ ] 2.1 Graficar loss/val_loss y accuracy/val_accuracy por época desde el historial de fit; verificar que ambas curvas distinguen entrenamiento y validación y usan la misma numeración de épocas.
- [ ] 2.2 Validar las series requeridas del historial y reportar claramente las ausentes; verificar con historiales completos e incompletos.

## 3. Predicciones individuales y aislamiento de test

- [ ] 3.1 Mostrar al menos 10 muestras de test distintas seleccionadas sin reemplazo con semilla fija 42, incluyendo clase real, clase predicha y probabilidad de la clase predicha; verificar cantidad, repetibilidad y valores de probabilidad en `[0, 1]`.
- [ ] 3.2 Si el test contiene entre 1 y 9 muestras, mostrar todas y advertir que no se alcanzó el mínimo de 10; verificar con conjuntos pequeños.
- [ ] 3.3 Integrar evaluación solo después de seleccionar/restaurar el mejor checkpoint por validación y evitar usar test para entrenamiento o callbacks; verificar en el flujo del notebook que la evaluación final ocurre una vez después del ajuste.
- [ ] 3.4 Documentar en el notebook las métricas, gráficas, predicciones y la función del test reservado; verificar que el flujo completo produce todos los resultados solicitados con la misma correspondencia de clases.
