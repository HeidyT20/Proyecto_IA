"""
verify_ethics.py - Script de Validacion de Sesgo y Fairness (Modulo E)
Proyecto Final IA-045 | Dominio: Retail / Logistica (Rossmann)

Calcula de forma empirica y en caliente las metricas de sesgo y equidad
sobre el conjunto de prueba para el reporte de docs/ethics_analysis.md.

Uso:
  python src/integration/verify_ethics.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Agregar raiz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    print("\n=======================================================")
    # ASCII Art simple para presentacion
    print("   MODULO E - VERIFICACION DE SESGO Y EQUIDAD (FAIRNESS)")
    print("=======================================================\n")

    # 1. Cargar datos originales
    data_path = "src/data/train_cleaned.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontro {data_path}. Ejecute primero: python src/ml/preprocess.py")
        
    print("[1/4] Cargando dataset de ventas y dividiendo split de prueba...")
    df = pd.read_csv(data_path)
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # Separar en train/test con la misma semilla que el pipeline para reproducibilidad
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Cargar scaler y transformar
    scaler_path = "src/ml/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontro {scaler_path}. Ejecute primero: python src/ml/train.py")
        
    print("[2/4] Cargando scaler y aplicando transformacion de features...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)

    # 3. Cargar modelo Random Forest
    modelo_path = "src/ml/modelo_rf.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontro {modelo_path}. Ejecute primero: python src/ml/train.py")
        
    print("[3/4] Cargando modelo Random Forest (Módulo B) entrenado...")
    with open(modelo_path, "rb") as f:
        rf_model = pickle.load(f)

    # 4. Predecir y agrupar para evaluar sesgos
    print("[4/4] Calculando errores absolutos por subgrupos y evaluando disparidad...\n")
    y_pred = rf_model.predict(X_test_scaled)
    
    results = X_test.copy()
    results['Actual_Sales'] = y_test
    results['Pred_Sales'] = y_pred
    results['Absolute_Error'] = np.abs(results['Actual_Sales'] - results['Pred_Sales'])
    
    global_mae = results['Absolute_Error'].mean()
    
    print("-" * 55)
    print(f" Métrica de Referencia: MAE Global = {global_mae:.2f}")
    print("-" * 55)
    
    # --- Sesgo por Promo ---
    print("\n>>> ANALISIS DE SESGO POR PROMOCION (PROMO) <<<")
    print("-" * 55)
    promo_groups = results.groupby('Promo')['Absolute_Error'].agg(['count', 'mean']).reset_index()
    promo_groups.columns = ['Promo Activa (0=No, 1=Si)', 'Cant. Registros', 'MAE del Subgrupo']
    print(promo_groups.to_string(index=False))
    
    mae_no_promo = promo_groups.iloc[0]['MAE del Subgrupo']
    mae_promo = promo_groups.iloc[1]['MAE del Subgrupo']
    disparidad = mae_promo / mae_no_promo
    print(f"\nRatio de disparidad (Con Promo / Sin Promo): {disparidad:.3f}x")
    print(f"-> Interpretacion: El modelo comete un {(disparidad - 1)*100:.1f}% MAS de error en dias de promocion.")
    print("-" * 55)
    
    # --- Sesgo por Store (Top 10 peor MAE) ---
    print("\n>>> ANALISIS DE SESGO POR TIENDA - TOP 10 PEOR MAE (Mas Perjudicadas) <<<")
    print("-" * 55)
    store_mae = results.groupby('Store')['Absolute_Error'].agg(['count', 'mean']).reset_index()
    store_mae.columns = ['Tienda (Store ID)', 'Cant. Registros', 'MAE por Tienda']
    store_mae_sorted = store_mae.sort_values(by='MAE por Tienda', ascending=False)
    
    top_peores = store_mae_sorted.head(10).copy()
    top_peores['Ratio vs Global'] = top_peores['MAE por Tienda'] / global_mae
    
    # Formatear la salida para que sea super legible
    pd.options.display.float_format = '{:,.2f}'.format
    print(top_peores.to_string(index=False))
    
    mean_peores = top_peores['MAE por Tienda'].mean()
    ratio_peores = mean_peores / global_mae
    print(f"\nRatio Promedio Top 10 Peores / Global: {ratio_peores:.2f}x")
    print(f"-> Las 10 tiendas mas perjudicadas tienen en promedio {ratio_peores:.2f} veces mas error que la media.")
    print("-" * 55)

    # --- Sesgo por Store (Top 5 mejor MAE) ---
    print("\n>>> ANALISIS DE SESGO POR TIENDA - TOP 5 MEJOR MAE (Mejor Predichas) <<<")
    print("-" * 55)
    top_mejores = store_mae_sorted.tail(5).copy()
    print(top_mejores.to_string(index=False))
    
    peor_mae = store_mae_sorted.iloc[0]['MAE por Tienda']
    mejor_mae = store_mae_sorted.iloc[-1]['MAE por Tienda']
    brecha = peor_mae / mejor_mae
    print(f"\nBrecha extrema (Peor Tienda / Mejor Tienda): {brecha:.1f}x")
    print(f"-> La peor tienda predicha tiene {brecha:.1f} veces mas error de inventario que la mejor.")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
