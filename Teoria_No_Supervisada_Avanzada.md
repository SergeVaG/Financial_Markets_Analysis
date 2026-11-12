# Aprendizaje No Supervisado: Marco Teórico Avanzado y Algoritmos Alternativos

El **aprendizaje no supervisado** es una rama crucial del Machine Learning donde los algoritmos analizan conjuntos de datos sin etiquetas previas ni respuestas de entrenamiento explícitas, buscando estructurar, segmentar o mapear la información a partir de sus propiedades intrínsecas.

---

## 1. El Paradigma Matemático y Probabilístico

En un entorno no supervisado, disponemos de un conjunto de datos sin etiquetas objetivo:

$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$$

Donde cada $\mathbf{x}_i \in \mathbb{R}^D$ es un vector multidimensional. El modelado puede enfocarse desde dos ópticas formales:

1.  **Enfoque Geométrico:** Encontrar una función de mapeo o partición $f: \mathbb{R}^D \rightarrow \mathcal{S}$ que maximice métricas de distancia inter-clúster y minimice distancias intra-clúster.
2.  **Enfoque Probabilístico:** Estimar la función de densidad de probabilidad subyacente $p(\mathbf{x})$. Por ejemplo, mediante un Modelo de Mezclas Gaussianas (GMM), asumimos que los datos se generan a partir de una combinación lineal de $K$ distribuciones normales:
    $$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
    Donde $\pi_k$ representa los pesos de mezcla (tales que $\sum \pi_k = 1$), y $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$ representan las medias y matrices de covarianza de cada componente.

---

## 2. Modelos de Agrupamiento Alternativos (Más allá de K-Means)

Cuando las agrupaciones en los datos reales no son esféricas ni tienen densidades uniformes, K-Means falla. Es necesario recurrir a estructuras de densidad o jerárquicas:

### A. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN agrupa puntos basándose en la densidad local del espacio de características, lo que le permite identificar formas arbitrarias y detectar valores atípicos (ruido) de manera nativa.

*   **Parámetros Críticos:**
    *   $\epsilon$ (Epsilon): El radio máximo que define la vecindad de un punto.
    *   $MinPts$: El número mínimo de puntos requeridos dentro de la vecindad $\epsilon$ para clasificar un punto como núcleo.
*   **Clasificación Matemática de Puntos:**
    *   **Punto Núcleo (Core Point):** Un punto $\mathbf{p}$ es un punto núcleo si su vecindad de radio $\epsilon$ contiene al menos $MinPts$ elementos:
        $$|N_\epsilon(\mathbf{p})| \ge MinPts \quad \text{donde} \quad N_\epsilon(\mathbf{p}) = \{\mathbf{q} \in \mathcal{D} \mid \|\mathbf{p} - \mathbf{q}\| \le \epsilon\}$$
    *   **Punto Frontera (Border Point):** No es un punto núcleo, pero es alcanzable por densidad desde algún punto núcleo.
    *   **Ruido / Outlier:** Cualquier punto que no es ni núcleo ni frontera.

### B. Agrupamiento Jerárquico Aglomerativo (AGNES)
Este enfoque construye una estructura arbórea de abajo hacia arriba (Bottom-Up), comenzando con cada observación como su propio clúster individual y fusionándolos iterativamente.

*   **Matriz de Distancia:** Se calcula inicialmente una matriz de distancias empíricas $D$ de tamaño $N \times N$.
*   **Criterios de Enlace (Linkage Criteria):** Definen matemáticamente la distancia entre dos clústeres $A$ y $B$:
    *   *Enlace Simple (Mínimo):* 
        $$d(A, B) = \min \{ \|\mathbf{a} - \mathbf{b}\| : \mathbf{a} \in A, \mathbf{b} \in B \}$$
    *   *Enlace Completo (Máximo):* 
        $$d(A, B) = \max \{ \|\mathbf{a} - \mathbf{b}\| : \mathbf{a} \in A, \mathbf{b} \in B \}$$
    *   *Enlace Promedio (Average):* 
        $$d(A, B) = \frac{1}{|A|\cdot|B|} \sum_{\mathbf{a} \in A} \sum_{\mathbf{b} \in B} \|\mathbf{a} - \mathbf{b}\|$$
    *   *Método de Ward:* Minimiza el incremento en la suma total de la varianza interna de los clústeres tras cada fusión consecutiva.

---

## 3. Validación Cruzada e Hiperparámetros en Aprendizaje No Supervisado

En ausencia de etiquetas verdaderas ($y$), el concepto tradicional de Validación Cruzada (Cross-Validation) debe reformularse. No se puede calcular un error de clasificación, pero se evalúa la estabilidad, generalización y consistencia del modelo ante perturbaciones del dataset.

### A. Validación por División de Datos (Train-Test Stability)
Para evitar el sobreajuste y comprobar la robustez de los clústeres encontrados:
1.  Se divide el conjunto de datos de manera aleatoria en un conjunto de entrenamiento $\mathcal{D}_{train}$ y uno de prueba $\mathcal{D}_{test}$.
2.  Se ajusta el modelo (ej. K-Means) en $\mathcal{D}_{train}$, obteniendo un conjunto de centroides $\mathcal{C}_{train}$.
3.  Se proyectan los datos de $\mathcal{D}_{test}$ sobre los centroides de entrenamiento para asignarles clústeres artificiales.
4.  Se compara la estructura de este agrupamiento proyectado frente a un modelo entrenado de manera independiente y directa sobre $\mathcal{D}_{test}$ utilizando métricas como el **Índice Rand Ajustado (ARI)**:
    $$ARI = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}}$$
    Un $ARI \approx 1$ certifica que la estructura descubierta es generalizable y estable.

### B. Análisis de Silueta Integrado
Para validar formalmente el número óptimo de clústeres $K$, la métrica de silueta calcula de forma individual para cada muestra $\mathbf{i}$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Donde:
*   $a(i)$ es la distancia media entre la muestra $\mathbf{i}$ y todos los demás puntos del mismo clúster.
*   $b(i)$ es la distancia media mínima desde la muestra $\mathbf{i}$ hacia todos los puntos del clúster vecino más cercano.

---

## 4. El Proceso Esencial de Normalización

En algoritmos no supervisados basados en distancias métricas, la escala de las variables altera los resultados de forma crítica. Si una característica opera en un rango numérico de magnitudes superiores, invalidará el peso de las demás variables en la norma geométrica.

La **Estandarización Z-Score** transforma el espacio de datos para que cada dimensión posea media cero ($\mu = 0$) y varianza unitaria ($\sigma = 1$):

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Donde:
$$\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij} \quad \text{y} \quad \sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_{ij} - \mu_j)^2}$$

Este paso garantiza que todas las métricas de comportamiento histórico aporten información de manera equilibrada y simétrica en la construcción del modelo.
