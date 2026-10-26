# Taller 10: Redes Neuronales Artificiales

**Inteligencia Artificial (ELP 8012), Universidad del Norte**
Profesor: Eduardo Zurek, Ph.D.

**Grupo:** Natalia Carpintero, Paula Núñez e Isabella Arrieta.

## Objetivo

Diseñar e implementar una **red neuronal artificial multicapa (MLP)** que clasifique los dígitos escritos a mano
del dataset `digits` de scikit-learn. Se comparan cuatro funciones de activación de la capa oculta
(`identity`, `logistic`, `tanh` y `relu`) y se determina cuál conviene usar, teniendo en cuenta también el
tiempo de entrenamiento.

## Contenido del repositorio

| Archivo | Descripción |
| --- | --- |
| `Taller10-Redes Neuronales Artificiales.ipynb` | Cuaderno con la solución: código, explicaciones y conclusiones. |
| `02_MLP_EZ_datos_sinteticos.ipynb` | Cuaderno de referencia del curso (MLP sobre datos sintéticos). |
| `_taller_10_Redes_Neuronales_Artificiales_parte_1.pdf` | Enunciado del taller. |

## Metodología

1. **Exploración del dataset**: 1797 imágenes de 8×8 píxeles (64 características, valores de 0 a 16) y
   10 clases casi balanceadas.
2. **Particiones estratificadas**: 60 % entrenamiento (1077), 20 % validación (360) y 20 % prueba (360).
3. **Modelo**: `Pipeline` con `StandardScaler` y `MLPClassifier` con una capa oculta de 64 neuronas,
   solver `lbfgs`, `alpha=1e-4`, `tol=1e-6`, `max_iter=2000` y semilla fija. Lo único que cambia entre los
   cuatro modelos es la función de activación.
4. **Métricas**: accuracy, balanced accuracy, F1 macro, log loss, número de iteraciones, convergencia y
   tiempo de entrenamiento. Como referencia mínima se usa un clasificador que siempre predice la clase mayoritaria.
5. **Estabilidad**: el entrenamiento se repite con 5 semillas de inicialización.
6. **Selección**: se elige la activación con mayor balanced accuracy en validación, con una regla fijada
   antes de mirar la prueba. El conjunto de prueba se evalúa una sola vez, al final.

## Resultados

Validación (media de 5 semillas):

| Activación | Accuracy val. | Iteraciones |
| --- | ---: | ---: |
| `identity` | 0,952 ± 0,005 | ~24 |
| `logistic` | **0,980 ± 0,005** | ~42 |
| `tanh` | 0,973 ± 0,002 | ~20 |
| `relu` | 0,969 ± 0,007 | ~22 |

Los cuatro modelos convergen y cada uno entrena en pocos segundos o menos. La activación
seleccionada fue **`logistic`**, que en el conjunto de prueba obtuvo **accuracy = 0,961** (14 errores de 360),
frente a 0,10 de la referencia trivial.

**Conclusión:** para este problema recomendamos `logistic`, porque obtiene la mayor precisión y el menor
log loss, y su costo extra de tiempo es despreciable en un conjunto de este tamaño. `tanh` es la mejor
alternativa si el tiempo de entrenamiento fuera una restricción. El análisis completo está en la sección 8
del cuaderno.

Los tiempos de entrenamiento dependen del equipo. Las precisiones se pueden reproducir porque las semillas
están fijas.

## Cómo ejecutarlo

Requisitos: Python 3.10 o superior con `numpy`, `pandas`, `matplotlib`, `scikit-learn` y Jupyter.

```bash
pip install numpy pandas matplotlib scikit-learn notebook
jupyter notebook "Taller10-Redes Neuronales Artificiales.ipynb"
```

Luego ejecuta todas las celdas (*Kernel → Restart & Run All*). Las versiones usadas fueron NumPy 2.4.4,
pandas 3.0.2 y scikit-learn 1.8.0.
