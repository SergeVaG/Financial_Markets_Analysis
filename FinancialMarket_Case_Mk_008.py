"""
Proyecto: Pipeline de Optimización de Parámetros para DBSCAN
Descripción: Implementa un gráfico de K-Distancias y un bucle de búsqueda en rejilla (Grid Search)
             para seleccionar automáticamente los valores óptimos de Epsilon (eps) y MinPts.
Idioma: Español
"""

import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# =====================================================================
# 1. INGENIERÍA DE CARACTERÍSTICAS (PROCESO PREVIO RESUMIDO)
# =====================================================================
print("📥 Cargando datos de muestra del NASDAQ...")
nasdaq_url = "https://nasdaqtrader.com"
req = urllib.request.Request(nasdaq_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    nasdaq_raw = pd.read_csv(response, sep="|")

sample_tickers = nasdaq_raw['Symbol'].dropna().unique().tolist()[:120]
sample_universe = [t.replace('.', '-') for t in sample_tickers] + ['^GSPC']

raw_data = yf.download(sample_universe, start="2024-01-01", end="2026-11-01", threads=True)
close_prices = raw_data['Close']

stock_prices = close_prices.drop(columns=['^GSPC']).dropna(thresh=int(len(close_prices)*0.95), axis=1)
market_returns = close_prices['^GSPC'].pct_change().dropna()
stock_returns = stock_prices.pct_change().dropna()

# Construcción de características estructurales
features = pd.DataFrame(index=stock_prices.columns)
features['Retorno_Anual'] = stock_returns.mean() * 252
features['Volatilidad'] = stock_returns.std() * np.sqrt(252)
market_var = market_returns.var()
features['Beta'] = [stock_returns[t].cov(market_returns) / market_var for t in features.index]
features['Momentum'] = (stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1

# Escalado Estándar (Z-Score) - Obligatorio para DBSCAN
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# =====================================================================
# 2. MÉTODO GRADIENTE: GRÁFICO DE K-DISTANCIAS (Para estimar Epsilon)
# =====================================================================
print("\n📈 Paso 1: Calculando Gráfico de K-Distancias...")
# Regla empírica: k de los vecinos más cercanos debe ser igual a MinPts esperado (ej. 4)
k_vecinos = 4
vecinos_modelo = NearestNeighbors(n_neighbors=k_vecinos)
vecinos_modelo.fit(scaled_features)
distancias, indices = vecinos_modelo.kneighbors(scaled_features)

# Ordenar las distancias del último vecino de menor a mayor
distancias_ordenadas = np.sort(distancias[:, k_vecinos - 1], axis=0)

plt.figure(figsize=(8, 4.5))
plt.plot(distancias_ordenadas, color='darkorange', linewidth=2)
plt.title('Gráfico de K-Distancias (Punto de Inflexión para Epsilon)', fontweight='bold')
plt.xlabel('Puntos de Datos Ordenados')
plt.ylabel(f'Distancia al Vecino Cercano K={k_vecinos}')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================================================
# 3. BUCLE DE BÚSQUEDA EN REJILLA (Grid Search Parameter Loop)
# =====================================================================
print("\n🔎 Paso 2: Ejecutando bucle de búsqueda en rejilla para hiperparámetros...")

# Definir los rangos de búsqueda basados en el gráfico previo y dimensiones
rango_eps = np.arange(0.3, 1.6, 0.2)
rango_min_pts = [3, 4, 5, 6]

resultados_grid = []

print(f"{'Eps':<6} | {'MinPts':<6} | {'Clústeres':<10} | {'% Ruido':<8} | {'Coef. Silueta':<13}")
print("-" * 55)

for eps in rango_eps:
    for min_pts in rango_min_pts:
        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        etiquetas = dbscan.fit_predict(scaled_features)
        
        # Calcular el número de clústeres detectados (excluyendo el ruido etiquetado como -1)
        n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
        
        # Calcular el porcentaje de valores atípicos (ruido)
        porcentaje_ruido = np.sum(etiquetas == -1) / len(etiquetas)
        
        # Calcular el Coeficiente de Silueta únicamente si existen al menos 2 clústeres válidos
        if n_clusters > 1:
            score_silueta = silhouette_score(scaled_features, etiquetas)
        else:
            score_silueta = -1.0  # Configuración por defecto si el modelo colapsa en un solo grupo
            
        resultados_grid.append({
            'Eps': round(eps, 2),
            'MinPts': min_pts,
            'Num_Clusters': n_clusters,
            'Pct_Ruido': round(porcentaje_ruido * 100, 2),
            'Silhouette': round(score_silueta, 4)
        })
        
        print(f"{eps:<6.2f} | {min_pts:<6} | {n_clusters:<10} | {porcentaje_ruido*100:<7.1f}% | {score_silueta:<13.4f}")

# Convertir la rejilla de datos en un DataFrame estructurado para análisis
df_grid = pd.DataFrame(resultados_grid)

# =====================================================================
# 4. EXTRACCIÓN DE LA CONFIGURACIÓN ÓPTIMA
# =====================================================================
# Filtro restrictivo: Buscamos maximizar Silueta, pero impidiendo que el ruido supere el 25% del mercado
filtros_seguros = df_grid[(df_grid['Num_Clusters'] > 1) & (df_grid['Pct_Ruido'] < 25.0)]

if not filtros_seguros.empty:
    mejor_configuracion = filtros_seguros.sort_values(by='Silhouette', ascending=False).iloc[0]
    print("\n" + "="*60)
    print("🏆 CONFIGURACIÓN DE PARÁMETROS ÓPTIMA ENCONTRADA")
    print(f"➡️ Epsilon (eps):     {mejor_configuracion['Eps']}")
    print(f"➡️ MinSamples (pts): {int(mejor_configuracion['MinPts'])}")
    print(f"➡️ Clústeres Libres: {int(mejor_configuracion['Num_Clusters'])}")
    print(f"➡️ Ruido Generado:   {mejor_configuracion['Pct_Ruido']}% del universo de acciones")
    print(f"➡️ Coef. de Silueta: {mejor_configuracion['Silhouette']}")
    print("="*60)
else:
    print("\n⚠️ Advertencia: No se encontró una combinación ideal estable sin generar ruido masivo.")
    print("Se sugiere ampliar el rango de búsqueda visualizando nuevamente el gráfico de K-Distancias.")
