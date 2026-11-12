# Aprendizaje Supervisado: Marco Teórico Fundacional y Algoritmos Core

El **aprendizaje supervisado** es el paradigma de Machine Learning donde un algoritmo aprende una función de mapeo a partir de datos de entrenamiento que contienen variables de entrada y sus respectivas etiquetas u objetivos correctos. A diferencia del aprendizaje no supervisado, el modelo cuenta con una "guía externa" que le permite evaluar explícitamente su error y corregir sus parámetros de forma iterativa.

---

## 1. El Paradigma Matemático y Estadístico

En el aprendizaje supervisado, el conjunto de datos de entrenamiento contiene pares ordenados de vectores de entrada y objetivos reales:

$$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$$

Donde cada $\mathbf{x}_i \in \mathbb{R}^D$ representa las características de una observación y $y_i$ es su etiqueta o valor real asociado. La meta fundamental es estimar una función predictiva $f(\mathbf{x}; \boldsymbol{\theta})$ parametrizada por un vector $\boldsymbol{\theta}$, tal que minimice una **Función de Pérdida o Coste** $\mathcal{L}$ sobre la distribución de los datos:

$$\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\big(y_i, f(\mathbf{x}_i; \boldsymbol{\theta})\big) + \lambda \mathcal{R}(\boldsymbol{\theta})$$

Donde $\mathcal{R}(\boldsymbol{\theta})$ representa el término de **Regularización** (L1/Lasso o L2/Ridge) controlado por el hiperparámetro $\lambda$, implementado para mitigar el sobreajuste (overfitting) mediante la penalización de la complejidad del vector de parámetros $\boldsymbol{\theta}$.

---

## 2. Clasificación vs. Regresión

El aprendizaje supervisado se divide categóricamente según la naturaleza de la variable objetivo $y$:

### A. Regresión
Se presenta cuando el objetivo $y$ es una variable **continua** y cuantitativa ($y \in \mathbb{R}$). 
*   *Métricas de Optimización:* Error Cuadrático Medio (MSE), Error Absoluto Medio (MAE), y el Coeficiente de Determinación ($R^2$).
*   *Objetivo:* Predecir magnitudes exactas (ej. el precio de cierre de una acción, la tasa de inflación o la rentabilidad esperada).

### B. Clasificación
Se presenta cuando el objetivo $y$ representa una variable **categórica** o cualitativa ($y \in \{0, 1\}$ para problemas binarios, o $y \in \{1, \dots, C\}$ para problemas multiclase).
*   *Métricas de Optimización:* Entropía Cruzada (Cross-Entropy Loss), Precisión (Accuracy), Precisión (Precision), Exhaustividad (Recall) y la puntuación F1-Score.
*   *Objetivo:* Asignar observaciones a categorías discretas (ej. decidir si una acción entrará en mercado bajista o alcista, o calificar el riesgo de impago de un crédito).

---

## 3. Algoritmos Fundacionales y su Mecánica

### Regresión Lineal y Logística (Modelos Paramétricos)
*   **Regresión Lineal:** Asume una relación lineal directa entre los datos de entrada y la salida continua:
    $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
*   **Regresión Logística:** Mapea la misma combinación lineal a una escala probabilística acotada entre 0 y 1 mediante el uso de la función de activación **Sigmoide**:
    $$p(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### Máquinas de Vectores de Soporte - SVM (Clasificación Geométrica)
Busca encontrar un hiperplano de decisión óptimo en un espacio multidimensional que separe las clases maximizando el **margen geométrico** entre los puntos más cercanos de cada categoría (llamados vectores de soporte).
*   **Truco del Kernel (Kernel Trick):** Cuando los datos no son linealmente separables en el espacio original, las SVM aplican una función kernel $K(\mathbf{x}_i, \mathbf{x}_j)$ para proyectar implícitamente los datos a un espacio de características de mayor dimensión donde la separación lineal sí es factible.

### Árboles de Decisión y Modelos de Ensamble (No Paramétricos)
*   **Árboles de Decisión:** Segmentan de forma recursiva el espacio de características mediante umbrales binarios optimizados. Utilizan criterios como la **Ganancia de Información** o la **Impureza de Gini** para maximizar la homogeneidad en los nodos hijos resultantes.
*   **Random Forest (Bagging):** Construye múltiples árboles de decisión independientes entrenados en subconjuntos aleatorios de los datos (Bootstrap) y promedia sus predicciones para reducir drásticamente la varianza matemática del modelo.
*   **Gradient Boosting (Boosting):** Entrena árboles de forma secuencial, donde cada nuevo árbol se optimiza específicamente para corregir los errores residuales (gradientes de la pérdida) cometidos por los árboles precedentes.

---

## 4. Validación Cruzada e Hiperparámetros (El Dilema Sesgo-Varianza)

A diferencia del paradigma no supervisado, la validación en modelos supervisados descansa sobre una base matemática rígida para estimar el error de generalización real ante datos nunca antes vistos.

### Validación Cruzada K-Fold (K-Fold Cross-Validation)
1.  El conjunto de datos original se permuta y fragmenta de manera equitativa en $K$ subgrupos disjuntos (folds).
2.  El modelo ejecuta un bucle iterativo de $K$ ciclos. En cada ciclo $k$, el fold $k$ se aísla de manera estricta para actuar como conjunto de validación o prueba, mientras que los restantes $K-1$ folds se consolidan para el entrenamiento.
3.  El rendimiento final estimado del modelo se reporta calculando la media estadistica y la desviación estándar de la métrica de rendimiento a lo largo de las $K$ evaluaciones consecutivas.

### El Compromiso Sesgo-Varianza (Bias-Variance Trade-off)
La capacidad de generalización de cualquier modelo supervisado está acotada por dos fuentes de error fundamentales:
*   **Sesgo (Bias):** Error introducido al aproximar un problema del mundo real mediante un modelo demasiado simple (ej. aproximar relaciones cúbicas con una Regresión Lineal). Produce un fenómeno conocido como **Subajuste (Underfitting)**.
*   **Varianza (Variance):** Error derivado de la extrema sensibilidad del modelo ante pequeñas fluctuaciones o ruido en el conjunto de entrenamiento (ej. un árbol de decisión sin límite de profundidad). Produce un fenómeno conocido como **Sobreajuste (Overfitting)**, donde el modelo memoriza el conjunto de datos de entrenamiento pero pierde su capacidad de generalizar ante datos externos.

---

## 5. El Proceso de Preprocesamiento y Fugas de Datos (Data Leakage)

El entrenamiento exitoso de modelos supervisados requiere un estricto aislamiento de los datos. Uno de los errores más comunes es la **Fuga de Datos (Data Leakage)**, la cual ocurre cuando información externa al conjunto de entrenamiento se filtra de manera inadvertida dentro del proceso de aprendizaje del modelo.

*   **Evitar el Escalado Global:** El escalado estándar Z-Score o MinMaxScaler jamás debe aplicarse sobre el conjunto de datos completo antes de realizar la partición. Parámetros como la media ($\mu$) y la desviación estándar ($\sigma$) deben calcularse **únicamente** utilizando los datos del subconjunto de entrenamiento, y posteriormente aplicarse de forma estricta sobre los datos de validación o prueba. De lo contrario, las métricas de evaluación del modelo estarán artificialmente infladas, ocultando un grave fallo de generalización estructural.
