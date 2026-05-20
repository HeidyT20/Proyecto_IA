"""
pipeline.py - Script de Integracion Completo (Modulo E)
Proyecto Final IA-045 | Dominio: Retail / Logistica (Rossmann)

Ejecuta el flujo completo de extremo a extremo:
  Datos -> ML (Random Forest) -> DL (Red Neuronal) -> NLP (Groq LLM) -> Resultado final

Uso:
  python src/integration/pipeline.py
  python src/integration/pipeline.py --no-api   (modo fallback sin Groq)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env si existe
load_dotenv()

# Agregar raiz del proyecto al path para importaciones relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def cargar_datos():
    """Carga el dataset limpio y divide en X/y de prueba."""
    from sklearn.model_selection import train_test_split

    data_path = "src/data/train_cleaned.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No se encontro {data_path}. "
            "Ejecute primero: python src/ml/preprocess.py"
        )

    print("[1/5] Cargando datos...")
    df = pd.read_csv(data_path)
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar con el scaler ya entrenado
    scaler_path = "src/ml/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No se encontro {scaler_path}. "
            "Ejecute primero: python src/ml/train.py"
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test, X_test


def evaluar_ml(X_test_scaled, y_test):
    """Carga y evalua el modelo Random Forest (Modulo B)."""
    from sklearn.metrics import mean_absolute_error, r2_score

    modelo_path = "src/ml/modelo_rf.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(
            f"No se encontro {modelo_path}. "
            "Ejecute primero: python src/ml/train.py"
        )

    print("[2/5] Evaluando modelo ML (Random Forest)...")
    with open(modelo_path, "rb") as f:
        rf_model = pickle.load(f)

    y_pred_rf = rf_model.predict(X_test_scaled)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"      RF  -> MAE: {mae_rf:.2f} | R2: {r2_rf:.4f}")
    return mae_rf, r2_rf, y_pred_rf


def evaluar_dl(X_test_scaled, y_test):
    """Carga y evalua la Red Neuronal (Modulo C)."""
    dl_pth = "src/dl/mejor_modelo_dl.pth"

    try:
        import torch
        from src.dl.model import SalesDeepRegressor

        if not os.path.exists(dl_pth):
            print("      DL  -> modelo no encontrado, omitiendo.")
            return None, None, None

        print("[3/5] Evaluando modelo DL (Red Neuronal PyTorch)...")
        model = SalesDeepRegressor(input_dim=8)
        model.load_state_dict(torch.load(dl_pth, map_location="cpu", weights_only=True))
        model.eval()

        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()

        from sklearn.metrics import mean_absolute_error, r2_score
        mae_dl = mean_absolute_error(y_test, preds)
        r2_dl = r2_score(y_test, preds)
        print(f"      DL  -> MAE: {mae_dl:.2f} | R2: {r2_dl:.4f}")
        return mae_dl, r2_dl, preds

    except Exception as e:
        print(f"      DL  -> Error al cargar modelo: {e}")
        return None, None, None


def ejecutar_nlp(X_test_raw, mae_rf, mae_dl, y_pred_rf, api_key=None):
    """Ejecuta el componente NLP (Modulo D): clasificacion + reporte."""
    from src.nlp.nlp_component import (
        sales_to_description,
        classify_with_llm,
        generate_sales_report,
        evaluate_report
    )

    print("[4/5] Ejecutando componente NLP (Groq LLM)...")

    # Generar descripcion de una muestra representativa
    muestra = X_test_raw.iloc[0]
    descripcion = sales_to_description(muestra)
    clasificacion = classify_with_llm(descripcion, api_key=api_key)

    print(f"      Descripcion: {descripcion}")
    print(f"      Clasificacion de demanda: {clasificacion}")

    # Generar reporte ejecutivo
    n_stores = int(X_test_raw["Store"].nunique()) if "Store" in X_test_raw.columns else "N/A"
    predictions_dict = {
        "rf_mae": round(mae_rf, 2) if mae_rf else "N/A",
        "dl_mae": round(mae_dl, 2) if mae_dl else "N/A",
        "avg_predicted": round(float(np.mean(y_pred_rf)), 0) if y_pred_rf is not None else "N/A",
        "period": "Validacion (20% del dataset)",
        "n_stores": n_stores,
    }

    report = generate_sales_report(predictions_dict, api_key=api_key)
    ok, issues = evaluate_report(report, predictions_dict)

    return report, ok, issues, clasificacion


def run_pipeline(api_key=None):
    """
    Funcion principal que ejecuta el pipeline completo.

    Flujo: Datos -> ML -> DL -> NLP -> Resultado

    Args:
        api_key: API key de Groq (opcional).

    Returns:
        Diccionario con todos los resultados del pipeline.
    """
    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETO - SISTEMA DE PREDICCION DE VENTAS")
    print("   Dominio: Retail / Logistica (Rossmann Store Sales)")
    print("=" * 55 + "\n")

    # Paso 1: Datos
    X_test_scaled, y_test, X_test_raw = cargar_datos()

    # Paso 2: ML
    mae_rf, r2_rf, y_pred_rf = evaluar_ml(X_test_scaled, y_test)

    # Paso 3: DL
    mae_dl, r2_dl, y_pred_dl = evaluar_dl(X_test_scaled, y_test)

    # Paso 4: CSP (Planificacion de Inventario)
    from src.search_csp.algorithm import backtracking
    from src.search_csp.agent import InventoryAgent
    print("[4/6] Ejecutando CSP (Planificacion de Inventario)...")
    
    # Seleccionar automáticamente el modelo con menor error (MAE)
    if mae_dl is not None and mae_dl < mae_rf:
        print(f"      [CSP] Seleccion automatica: Modelo DL (Red Neuronal) elegido por menor MAE ({mae_dl:.2f} vs {mae_rf:.2f})")
        predicciones_elegidas = y_pred_dl
        modelo_elegido_nombre = "Red Neuronal (DL)"
    else:
        comparativa_dl = f"{mae_dl:.2f}" if mae_dl is not None else "N/A"
        print(f"      [CSP] Seleccion automatica: Modelo ML (Random Forest) elegido por menor MAE ({mae_rf:.2f} vs {comparativa_dl})")
        predicciones_elegidas = y_pred_rf
        modelo_elegido_nombre = "Random Forest (ML)"
    
    # Simular demanda para 3 productos clave basada en el promedio predicho por el modelo ganador
    avg_demand = int(np.mean(predicciones_elegidas)) if predicciones_elegidas is not None else 5000
    predicted_demand = {"A": int(avg_demand * 0.4), "B": int(avg_demand * 0.3), "C": int(avg_demand * 0.3)}
    
    initial_state = {
        "inventario": {"A": 1000, "B": 800, "C": 1500},
        "demanda": predicted_demand,
        "presupuesto": 250000,
    }
    productos = ["A", "B", "C"]
    costos = {"A": 50, "B": 100, "C": 30}
    capacidad = 5000
    
    agent = InventoryAgent(productos, costos, capacidad)
    final_state = backtracking(initial_state, productos, costos, capacidad)
    
    orders = {}
    if final_state:
        for p in productos:
            orders[p] = final_state["inventario"][p] - initial_state["inventario"][p]
        print(f"      CSP -> Ordenes generadas: {orders}")
    else:
        print("      CSP -> No se encontro solucion factible.")

    # Paso 5: NLP
    print("[5/6] Ejecutando componente NLP (Groq LLM)...")
    from src.nlp.nlp_component import sales_to_description, classify_with_llm, generate_sales_report, evaluate_report
    
    muestra = X_test_raw.iloc[0] if len(X_test_raw) > 0 else pd.Series()
    descripcion = sales_to_description(muestra)
    clasificacion = classify_with_llm(descripcion, api_key=api_key)
    
    n_stores = int(X_test_raw["Store"].nunique()) if "Store" in X_test_raw.columns else "N/A"
    predictions_dict = {
        "rf_mae": round(mae_rf, 2) if mae_rf else "N/A",
        "dl_mae": round(mae_dl, 2) if mae_dl else "N/A",
        "avg_predicted": round(float(np.mean(predicciones_elegidas)), 0) if predicciones_elegidas is not None else "N/A",
        "period": "Validacion (20% del dataset)",
        "n_stores": n_stores,
        "orders": orders,
        "csp_budget_remaining": final_state["presupuesto"] if final_state else "N/A",
        "modelo_elegido": modelo_elegido_nombre
    }
    
    report = generate_sales_report(predictions_dict, api_key=api_key)
    ok, issues = evaluate_report(report, predictions_dict)

    # Paso 6: Resultado final
    print("\n[6/6] Resultado Final")
    print("=" * 55)
    print("\n" + report)

    if not ok:
        print("\nAvisos del reporte:")
        for issue in issues:
            print(f"  - {issue}")

    resultado = {
        "rf_mae": mae_rf,
        "rf_r2": r2_rf,
        "dl_mae": mae_dl,
        "dl_r2": r2_dl,
        "clasificacion_nlp": clasificacion,
        "report": report,
        "report_ok": ok,
        "report_issues": issues,
    }

    print("\n" + "=" * 55)
    print("Pipeline completado exitosamente.")
    print("=" * 55 + "\n")

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de IA - Prediccion de Ventas Rossmann")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Ejecutar en modo fallback sin API de Groq"
    )
    args = parser.parse_args()

    api_key = "" if args.no_api else os.getenv("GROQ_API_KEY", "")
    run_pipeline(api_key=api_key)
