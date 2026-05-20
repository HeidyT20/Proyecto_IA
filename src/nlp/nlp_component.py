"""
nlp_component.py - Modulo D: NLP / LLM
Proyecto Final IA-045 | Dominio: Retail / Logistica (Rossmann)

Implementa tres enfoques de NLP para clasificacion de contexto de ventas,
siguiendo el patron de comparativa del Lab 8 (LLMs Comparativa):

  Enfoque A (descartado) : TF-IDF + Logistic Regression
  Enfoque B (descartado) : Sentence Embeddings (all-MiniLM-L6-v2)
  Enfoque C (elegido)    : Groq LLM generativo (llama-3.1-8b-instant)

Tarea A - Clasificacion: Alta Demanda / Demanda Normal / Baja Demanda
Tarea B - Generacion:    Resumen ejecutivo de ventas en lenguaje natural

Uso sin API key: el modulo funciona en modo fallback (plantilla estructurada).
Uso con API key: configurar variable de entorno GROQ_API_KEY.
"""

import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env si existe
load_dotenv()

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 0. CONSTANTES Y PREPROCESAMIENTO DE TEXTO
# ─────────────────────────────────────────────

DAY_NAMES = {
    1: "lunes", 2: "martes", 3: "miercoles",
    4: "jueves", 5: "viernes", 6: "sabado", 7: "domingo"
}

LABEL_NAMES = ["Baja Demanda", "Demanda Normal", "Alta Demanda"]


def preprocess_text(text: str) -> str:
    """Normaliza texto: minusculas, sin espacios extras."""
    return " ".join(text.strip().lower().split())


def sales_to_description(row: pd.Series) -> str:
    """
    Convierte una fila numerica del dataset Rossmann en una descripcion textual.
    Este preprocesamiento es el puente entre los datos numericos y el componente NLP.

    Ejemplo de salida:
      "Tienda 5, lunes, con promocion activa, dia normal, periodo lectivo.
       Clientes estimados: 800."
    """
    day = DAY_NAMES.get(int(row.get("DayOfWeek", 1)), "dia desconocido")
    promo = "con promocion activa" if int(row.get("Promo", 0)) == 1 else "sin promocion"
    holiday_val = str(row.get("StateHoliday", "0"))
    holiday = "festivo estatal" if holiday_val != "0" else "dia normal"
    school = "periodo vacacional" if int(row.get("SchoolHoliday", 0)) == 1 else "periodo lectivo"
    store = int(row.get("Store", 0))
    customers = int(row.get("Customers", 0))

    return (
        f"Tienda {store}, {day}, {promo}, {holiday}, {school}. "
        f"Clientes estimados: {customers}."
    )


def generate_labeled_dataset(df: pd.DataFrame, n_samples: int = 300) -> Tuple[List[str], List[int]]:
    """
    Genera textos etiquetados para clasificacion de demanda.

    Etiquetas basadas en percentiles de ventas del dataset:
      0 = Baja Demanda  (ventas < percentil 33)
      1 = Demanda Normal (percentil 33 <= ventas < percentil 66)
      2 = Alta Demanda  (ventas >= percentil 66)

    Args:
        df: DataFrame del dataset Rossmann limpio.
        n_samples: Numero de muestras a generar (balanceadas por clase).

    Returns:
        Tuple de (textos, etiquetas_numericas)
    """
    p33 = df["Sales"].quantile(0.33)
    p66 = df["Sales"].quantile(0.66)

    df = df.copy()
    df["_label"] = 1  # Demanda Normal por defecto
    df.loc[df["Sales"] < p33, "_label"] = 0
    df.loc[df["Sales"] >= p66, "_label"] = 2

    # Balancear clases
    samples_per_class = n_samples // 3
    frames = []
    for lbl in [0, 1, 2]:
        subset = df[df["_label"] == lbl]
        n = min(len(subset), samples_per_class)
        frames.append(subset.sample(n=n, random_state=42))

    df_balanced = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

    texts = [sales_to_description(row) for _, row in df_balanced.iterrows()]
    labels = df_balanced["_label"].tolist()

    return texts, labels


# ─────────────────────────────────────────────
# ENFOQUE A — TF-IDF + Logistic Regression
# (Alternativa descartada — ver nlp_decisions.md)
# ─────────────────────────────────────────────

def train_tfidf_classifier(texts: List[str], labels: List[int]):
    """
    Entrena un clasificador TF-IDF + Logistic Regression.

    Descartado porque: no reconoce sinonimos ni variaciones semanticas.
    Por ejemplo, "dia festivo" y "festivo estatal" son tratados como terminos
    distintos aunque signifiquen lo mismo para el dominio.

    Returns:
        Tuple de (vectorizer, clf) entrenados.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
        analyzer="word"
    )

    X = vectorizer.fit_transform([preprocess_text(t) for t in texts])
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X, labels)
    return vectorizer, clf


def classify_tfidf(text: str, vectorizer, clf) -> str:
    """Clasifica un texto usando TF-IDF. Retorna etiqueta de demanda."""
    X = vectorizer.transform([preprocess_text(text)])
    pred = clf.predict(X)[0]
    return LABEL_NAMES[pred]


# ─────────────────────────────────────────────
# ENFOQUE B — Sentence Embeddings
# (Alternativa descartada — ver nlp_decisions.md)
# ─────────────────────────────────────────────

def train_embedding_classifier(texts: List[str], labels: List[int]):
    """
    Entrena un clasificador con Sentence Embeddings (all-MiniLM-L6-v2).

    Descartado porque: aunque supera a TF-IDF semanticamente, no puede
    GENERAR texto nuevo. Para el dominio Rossmann necesitamos generar
    resumenes ejecutivos, no solo clasificar. El LLM generativo cumple
    ambas tareas simultaneamente.

    Returns:
        Tuple de (modelo_embeddings, clf) entrenados.
    """
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = modelo.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(embeddings, labels)
    return modelo, clf


def classify_embedding(text: str, modelo, clf) -> str:
    """Clasifica un texto usando embeddings semanticos. Retorna etiqueta de demanda."""
    emb = modelo.encode([preprocess_text(text)], convert_to_numpy=True)
    pred = clf.predict(emb)[0]
    return LABEL_NAMES[pred]


# ─────────────────────────────────────────────
# ENFOQUE C — Groq LLM (ELEGIDO)
# Razon: unico enfoque que clasifica Y genera texto nuevo
# ─────────────────────────────────────────────

def classify_with_llm(description: str, api_key: Optional[str] = None) -> str:
    """
    Clasifica el contexto de ventas usando Groq LLM (Tarea A).

    Elegido porque combina clasificacion semantica + capacidad generativa.
    Modelo: llama-3.1-8b-instant (gratuito, ~60ms latencia).

    Args:
        description: Texto generado por sales_to_description().
        api_key: API key de Groq. Si None, usa variable de entorno GROQ_API_KEY.

    Returns:
        Una de: 'Alta Demanda', 'Demanda Normal', 'Baja Demanda'
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        # Fallback deterministico: analisis de palabras clave
        text_lower = description.lower()
        if "promocion activa" in text_lower or "festivo" in text_lower:
            return "Alta Demanda"
        elif "sin promocion" in text_lower and "dia normal" in text_lower:
            return "Baja Demanda"
        return "Demanda Normal"

    try:
    
        # pyrefly: ignore [missing-import]
        from groq import Groq

        client = Groq(api_key=key)
        prompt = (
            "Clasifica el siguiente contexto de ventas de una tienda retail "
            "en EXACTAMENTE una de estas tres categorias: "
            "'Alta Demanda', 'Demanda Normal', o 'Baja Demanda'.\n"
            "Responde SOLO con el nombre de la categoria, sin puntos ni explicacion.\n\n"
            f"Contexto: {description}"
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        # Normalizar respuesta del LLM
        for label in LABEL_NAMES:
            if label.lower() in result.lower():
                return label
        return result

    except Exception as e:
        return f"Error LLM: {e}"


def generate_sales_report(predictions: Dict, api_key: Optional[str] = None) -> str:
    """
    Genera un resumen ejecutivo de ventas en lenguaje natural (Tarea B).

    Usa el LLM de Groq si hay API key disponible.
    Fallback: plantilla estructurada si no hay conexion.

    Args:
        predictions: Diccionario con metricas del pipeline ML/DL.
        api_key: API key de Groq (opcional, usa variable de entorno si no se pasa).

    Returns:
        Resumen ejecutivo como string.
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")

    context_lines = [
        f"- Modelo ML (Random Forest) MAE: {predictions.get('rf_mae', 'N/A')}",
        f"- Modelo DL (Red Neuronal) MAE: {predictions.get('dl_mae', 'N/A')}",
        f"- Modelo seleccionado automáticamente (menor MAE): {predictions.get('modelo_elegido', 'N/A')}",
        f"- Ventas promedio predichas por el modelo seleccionado: {predictions.get('avg_predicted', 'N/A')} unidades",
        f"- Ordenes planificadas (CSP): {predictions.get('orders', 'N/A')}",
        f"- Presupuesto restante (CSP): {predictions.get('csp_budget_remaining', 'N/A')}",
        f"- Periodo analizado: {predictions.get('period', 'reciente')}",
        f"- Tiendas evaluadas: {predictions.get('n_stores', 'N/A')}",
    ]
    context = "\n".join(context_lines)

    if key:
        try:
            print("      [NLP] API Key detectada. Consultando Llama 3.3 70B en Groq...")
            # type: ignore
            # pyright: ignore [missing-import]
            from groq import Groq

            client = Groq(api_key=key)
            prompt = (
                "Eres un analista senior de ventas retail. "
                "Genera un resumen ejecutivo en español de 3 a 4 oraciones "
                "basado ÚNICAMENTE en las siguientes predicciones generadas por modelos de Inteligencia Artificial.\n"
                "REGLAS ESTRUCTURALES Y DE SEGURIDAD:\n"
                "1. Está terminantemente PROHIBIDO sugerir o inventar incrementos o decrementos de inventario en forma de porcentajes abstractos (por ejemplo, NO digas cosas como 'ajustar en un 10%' o 'un 5%').\n"
                "2. Tus conclusiones y recomendaciones de inventario deben basarse estrictamente en las cantidades exactas de unidades absolutas a ordenar calculadas por el CSP (que se detallan en el campo 'Ordenes planificadas'). Cita estas cantidades textuales.\n"
                "3. Debes citar explícitamente los valores de error medio absoluto (MAE) de los modelos ML y DL provistos en los datos para fundamentar la precisión del análisis.\n"
                "4. Mantén un tono sumamente analítico y objetivo, apegado estrictamente al contexto empírico provisto.\n\n"
                f"Predicciones del sistema:\n{context}"
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"      [NLP] Error conectando a Groq: {e}")
            print("      [NLP] Activando degradacion elegante (Fallback)...")
            pass  # Graceful degradation al fallback
    else:
        print("      [NLP] API Key no encontrada. Usando modo Fallback (plantilla local)...")

    # Fallback: plantilla estructurada (funciona sin API)
    return (
        "=== Resumen Ejecutivo de Predicciones de Ventas ===\n\n"
        f"{context}\n\n"
        "Conclusiones:\n"
        f"  - El modelo Random Forest obtuvo MAE = {predictions.get('rf_mae', 'N/A')}, "
        "siendo el mas preciso para este dominio tabular.\n"
        f"  - La Red Neuronal (DL) obtuvo MAE = {predictions.get('dl_mae', 'N/A')}, "
        "mostrando que datos tabulares favorecen modelos basados en arboles.\n"
        f"  - El modulo CSP genero exitosamente ordenes de reabastecimiento: {predictions.get('orders', 'N/A')}\n"
        "Recomendacion: Usar las predicciones del Random Forest para ajuste de inventario. "
        "Priorizar reabastecimiento en tiendas con dias de promocion y festivos estatales."
    )


def evaluate_report(report: str, predictions: Dict) -> Tuple[bool, List[str]]:
    """
    Evalua que el reporte generado mencione los datos clave del plan.

    Returns:
        Tuple (ok: bool, lista_de_problemas: List[str])
    """
    issues = []
    normalized = preprocess_text(report)

    if "mae" not in normalized and "error" not in normalized:
        issues.append("El reporte no menciona la metrica de error (MAE)")
    if "ventas" not in normalized and "venta" not in normalized and "sales" not in normalized:
        issues.append("El reporte no menciona ventas")
    if "random forest" not in normalized and "modelo" not in normalized:
        issues.append("El reporte no referencia el modelo utilizado")

    return len(issues) == 0, issues


# ─────────────────────────────────────────────
# MAIN — Demo rapida sin API key
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== NLP Component Demo (modo fallback sin API key) ===\n")

    # Generar descripcion de ejemplo
    ejemplo_row = pd.Series({
        "Store": 5, "DayOfWeek": 1, "Promo": 1,
        "StateHoliday": "0", "SchoolHoliday": 0, "Customers": 850
    })
    desc = sales_to_description(ejemplo_row)
    print(f"Descripcion generada:\n  {desc}\n")

    # Clasificacion con fallback (sin API)
    clasificacion = classify_with_llm(desc, api_key="")
    print(f"Clasificacion (fallback): {clasificacion}\n")

    # Reporte con fallback
    predictions_demo = {
        "rf_mae": 1697.55,
        "dl_mae": 1987.30,
        "avg_predicted": 7200,
        "period": "Mayo 2015",
        "n_stores": 50
    }
    report = generate_sales_report(predictions_demo, api_key="")
    print("Reporte generado (fallback):")
    print(report)
