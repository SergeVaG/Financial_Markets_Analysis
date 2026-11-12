"""
Proyecto: Pipeline de Clustering K-Means para Acciones de NASDAQ
Descripción: Descarga el listado oficial de NASDAQ, genera 5 características avanzadas,
             ejecuta el método del codo, entrena el modelo final con K=4 y grafica las combinaciones.
Idioma: Español
"""

import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2026-11-01"
K_OPTIMO = 4
TAMANO_MUESTRA = 150  # Salvaguarda para evitar bloqueos por parte de Yahoo Finance

def obtener_tickers_nasdaq():
    """Descarga el directorio oficial en vivo de las acciones listadas en NASDAQ."""
    print("📥 Paso 1: Descargando el directorio completo y en vivo de tickers de NASDAQ...")
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    
    req = urllib.request.Request(nasdaq_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        nasdaq_raw = pd.read_csv(response, sep="|")
        
    nasdaq_raw = nasdaq_raw.iloc[:-1]  # Elimina el pie de página con la marca de tiempo
    tickers_completos = nasdaq_raw['Symbol'].dropna().unique().tolist()
    print(f"✅ ¡Éxito! Se encontraron {len(tickers_completos)} listados activos en la bolsa.")
    return tickers_completos

def descargar_datos_historicos(tickers, tamano_muestra):
    """Descarga los precios históricos y realiza la limpieza inicial de datos."""
    universo_muestra = [t.replace('.', '-') for t in tickers[:tamano_muestra]]
    lista_descarga = universo_muestra + ['^GSPC']
    
    print(f"\n📥 Paso 2: Extrayendo matriz de datos históricos para {tamano_muestra} acciones...")
    datos_raw = yf.download(lista_descarga, start=FECHA_INICIO, end=FECHA_FIN, threads=True)
    precios_cierre = datos_raw['Close']
    
    precios_acciones = precios_cierre[universo_muestra]
    precios_mercado = precios_cierre['^GSPC']
    
    # Limpieza: Filtrar acciones que tengan más del 5% de datos faltantes
    umbral_faltantes = 0.05 * len(precios_cierre)
    precios_acciones = precios_acciones.dropna(thresh=len(precios_cierre) - umbral_faltantes, axis=1)
    tickers_activos = precios_acciones.columns.tolist()
    
    retornos_acciones = precios_acciones.pct_change().dropna()
    retornos_mercado = precios_mercado.pct_change().dropna()
    
    return precios_acciones, retornos_acciones, retornos_mercado, tickers_activos

def calcular_caracteristicas(precios_acciones, retornos_acciones, retornos_mercado, tickers_activos):
    """Calcula las 5 métricas de comportamiento financiero para el clustering."""
    print("⚙️ Paso 3: Calculando las 5 características conductuales...")
    df_features = pd.DataFrame(index=tickers_activos)
    
    # 1. Rendimiento Anualizado
    df_features['Retorno_Anual'] = retornos_acciones.mean() * 252
    
    # 2. Volatilidad Anualizada
    df_features['Volatilidad'] = retornos_acciones.std() * np.sqrt(252)
    
    # 3. Beta de Mercado
    varianza_mercado = retornos_mercado.var()
    df_features['Beta'] = [retornos_acciones[t].cov(retornos_mercado) / varianza_mercado for t in tickers_activos]
    
    # 4. Momentum Estructural
    df_features['Momentum'] = (precios_acciones.iloc[-1] / precios_acciones.iloc[0]) - 1
    
    # 5. Máximo Drawdown (Peor caída pico a valle)
    lista_max_dd = []
    for t in tickers_activos:
        indice_riqueza = (1 + retornos_acciones[t]).cumprod()
        lista_max_dd.append(((indice_riqueza - indice_riqueza.cummax()) / indice_riqueza.cummax()).min())
    df_features['Max_Drawdown'] = lista_max_dd
    
    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
    return df_features

def graficar_metodo_codo(scaled_features):
    """Calcula e imprime el gráfico de optimización para el método del codo."""
    print("\n📊 Calculando optimización de clústeres (Método del Codo)...")
    wcss = []
    rango_k = range(1, 10)
    for k in rango_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 4.5))
    plt.plot(rango_k, wcss, marker='s', linestyle='-', color='indigo', linewidth=2)
    plt.title('Método del Codo: Optimización de Clústeres en NASDAQ', fontweight='bold')
    plt.xlabel('Número de Clústeres (K)')
    plt.ylabel('Inercia / WCSS')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def graficar_proyecciones_2d(df_features):
    """Genera una cuadrícula de gráficos combinando las características en pares 2D."""
    columnas_features = ['Retorno_Anual', 'Volatilidad', 'Beta', 'Momentum', 'Max_Drawdown']
    pares_features = list(itertools.combinations(columnas_features, 2))
    
    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    axes = axes.flatten()
    
    for i, (col_x, col_y) in enumerate(pares_features):
        ax = axes[i]
        
        # Puntos coloreados según la etiqueta global de su clúster
        scatter = ax.scatter(df_features[col_x], df_features[col_y], 
                             c=df_features['Cluster'], cmap='plasma', s=150, alpha=0.8, edgecolors='black')
        
        # Añadir etiquetas de texto con los tickers de las acciones
        for ticker in df_features.index:
            ax.annotate(ticker, (df_features.loc[ticker, col_x], df_features.loc[ticker, col_y]),
                        textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='semibold')
            
        ax.set_title(f'{col_y} vs {col_x}', fontsize=11, fontweight='bold')
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    for j in range(len(pares_features), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Clústeres de Acciones Mostrados en Proyecciones Combinadas 2D', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

# =====================================================================
# FLUJO PRINCIPAL DE EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    # Fase 1: Extracción del listado de la bolsa
    tickers_completos = obtener_tickers_nasdaq()
    
    # Fase 2: Descarga de precios
    p_acciones, r_acciones, r_mercado, tickers_filtrados = descargar_datos_historicos(tickers_completos, TAMANO_MUESTRA)
    
    # Fase 3: Procesamiento de datos y métricas financieras
    features_df = calcular_caracteristicas(p_acciones, r_acciones, r_mercado, tickers_filtrados)
    
    # Fase 4: Escalado estándar Z-Score
    scaler = StandardScaler()
    caracteristicas_escaladas = scaler.fit_transform(features_df)
    
    # Fase 5: Visualización del método del codo
    graficar_metodo_codo(caracteristicas_escaladas)
    
    # Fase 6: Entrenamiento y ajuste del modelo K-Means final (K = 4)
    print(f"\n🎯 Ajustando modelo de clustering definitivo (K = {K_OPTIMO})...")
    kmeans_final = KMeans(n_clusters=K_OPTIMO, init='k-means++', random_state=42, n_init='auto')
    features_df['Cluster'] = kmeans_final.fit_predict(caracteristicas_escaladas)
    
    # Fase 7: Filtrado del clúster con mejor desempeño (Filtro por Ratio de Sharpe)
    rendimiento_clusters = features_df.groupby('Cluster').agg({
        'Retorno_Anual': 'mean',
        'Volatilidad': 'mean'
    })
    rendimiento_clusters['Sharpe_Grupal'] = rendimiento_clusters['Retorno_Anual'] / rendimiento_clusters['Volatilidad']
    
    id_mejor_cluster = rendimiento_clusters['Sharpe_Grupal'].idxmax()
    acciones_elite = features_df[features_df['Cluster'] == id_mejor_cluster]
    
    # Impresión de Resultados por Consola
    print("\n" + "="*55)
    print(f"🌟 CLÚSTER CON MEJOR RENDIMIENTO ENCONTRADO: Grupo {id_mejor_cluster}")
    print(f"📊 Métricas Promedio del Clúster: Retorno: {rendimiento_clusters.loc[id_mejor_cluster, 'Retorno_Anual']:.2%}, Volatilidad: {rendimiento_clusters.loc[id_mejor_cluster, 'Volatilidad']:.2%}")
    print(f"📋 Total de Tickers dentro de este Clúster: {len(acciones_elite)}")
    print("="*55)
    
    print("\n✨ Tickers (Símbolos) en el Clúster de Alto Rendimiento:")
    print(acciones_elite.index.tolist())
    
    print("\n🔎 Desglose Analítico Detallado del Clúster de Élite:")
    print(acciones_elite.round(4))
    
    # Fase 8: Mostrar la matriz de gráficos combinados 2D
    print("\n🖼️ Generando matriz de gráficos de dispersión multidimensionales...")
    graficar_proyecciones_2d(features_df)
