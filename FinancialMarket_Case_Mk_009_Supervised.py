"""
Proyecto: Pipeline Supervisado para Acciones (Clasificación y Regresión)
Descripción: Predice de forma simultánea si una acción subirá/bajará mañana (Clasificación)
             y estimará el precio de cierre continuo futuro (Regresión).
Idioma: Español
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# =====================================================================
# 1. ADQUISICIÓN DE DATOS Y INGENIERÍA DE CARACTERÍSTICAS
# =====================================================================
print("📥 Descargando datos históricos de la acción objetivo (Ejemplo: AAPL)...")
ticker = "AAPL"
df = yf.download(ticker, start="2024-01-01", end="2026-11-01")

# Extraer el vector de precios de cierre
df_stock = pd.DataFrame(index=df.index)
df_stock['Precio_Cierre'] = df['Close']

print("⚙️ Generando indicadores técnicos (Variables de entrada / X)...")
# Característica 1: Rendimiento diario histórico
df_stock['Retorno_Diario'] = df_stock['Precio_Cierre'].pct_change()

# Característica 2: Volatilidad rodante de 5 días
df_stock['Volatilidad_5D'] = df_stock['Retorno_Diario'].rolling(window=5).std()

# Característica 3: Diferencial del cruce de medias móviles (SMA 5 vs SMA 20)
df_stock['SMA_5'] = df_stock['Precio_Cierre'].rolling(window=5).mean()
df_stock['SMA_20'] = df_stock['Precio_Cierre'].rolling(window=20).mean()
df_stock['Cruce_SMA'] = df_stock['SMA_5'] - df_stock['SMA_20']

# =====================================================================
# 2. DEFINICIÓN DE OBJETIVOS SUPERVISADOS (Y)
# =====================================================================
# Objetivo de Regresión: El precio de cierre del día de MAÑANA (Continuo)
df_stock['Objetivo_Precio'] = df_stock['Precio_Cierre'].shift(-1)

# Objetivo de Clasificación: ¿El precio de mañana será mayor al de hoy? (Binario: 1 = Sube, 0 = Baja)
df_stock['Objetivo_Direccion'] = np.where(df_stock['Objetivo_Precio'] > df_stock['Precio_Cierre'], 1, 0)

# Limpiar las filas residuales creadas por las ventanas rodantes y el desfasado (shift)
df_modelos = df_stock[['Volatilidad_5D', 'Cruce_SMA', 'Retorno_Diario', 'Objetivo_Precio', 'Objetivo_Direccion']].dropna()

# Separar características (X) de los vectores objetivos (y)
columnas_X = ['Volatilidad_5D', 'Cruce_SMA', 'Retorno_Diario']
X = df_modelos[columnas_X]
y_regresion = df_modelos['Objetivo_Precio']
y_clasificacion = df_modelos['Objetivo_Direccion']

# =====================================================================
# 3. DIVISIÓN TEMPORAL Y CONTROL DE FUGAS DE DATOS (Data Leakage)
# =====================================================================
# Dividimos de forma secuencial (sin barajar) debido a la naturaleza cronológica de las acciones
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regresion, test_split=0.2, shuffle=False)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clasificacion, test_split=0.2, shuffle=False)

print(f"📋 Tamaño del conjunto de Entrenamiento: {len(X_train)} días")
print(f"📋 Tamaño del conjunto de Prueba / Validación: {len(X_test)} días")

# ESCALADO PREVENTIVO: Calculamos media y desviación SOLO con el conjunto de entrenamiento
scaler = StandardScaler()
X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)  # Se utiliza el mismo ajuste de Train para evitar filtraciones

# =====================================================================
# 4. MODELO A: CLASIFICACIÓN (¿Sube o Baja?)
# =====================================================================
print("\n🤖 Entrenando Modelo de Clasificación (Random Forest Classifier)...")
modelo_clasificador = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_clasificador.fit(X_train_escalado, y_train_clf)

predicciones_clf = modelo_clasificador.predict(X_test_escalado)

print("\n📊 REPORTE DE RENDIMIENTO (CLASIFICACIÓN BINARIA):")
print(classification_report(y_test_clf, predicciones_clf, target_names=["Baja (0)", "Sube (1)"]))

# =====================================================================
# 5. MODELO B: REGRESIÓN (Precio Continuo)
# =====================================================================
print("\n🤖 Entrenando Modelo de Regresión (Random Forest Regressor)...")
modelo_regresor = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_regresor.fit(X_train_escalado, y_train_reg)

predicciones_reg = modelo_regresor.predict(X_test_escalado)

mse = mean_squared_error(y_test_reg, predicciones_reg)
r2 = r2_score(y_test_reg, predicciones_reg)

print("\n📊 MÉTRICAS DE RENDIMIENTO (REGRESIÓN CONTINUA):")
print(f"➡️ Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"➡️ Coeficiente de Determinación (R²): {r2:.4f}")

# =====================================================================
# 6. VISUALIZACIÓN DE RESULTADOS CONTINUOS
# =====================================================================
plt.figure(figsize=(12, 6))
plt.plot(y_test_reg.index, y_test_reg.values, label="Precio Real Objetivo", color="black", linewidth=1.5)
plt.plot(y_test_reg.index, predicciones_reg, label="Predicción del Modelo", color="crimson", linestyle="--", linewidth=1.5)
plt.title(f"Modelo Supervisado de Regresión: Predicción de Precios para {ticker}", fontsize=14, fontweight="bold")
plt.xlabel("Línea Temporal (Conjunto de Prueba)")
plt.ylabel("Precio de la Acción (USD)")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
