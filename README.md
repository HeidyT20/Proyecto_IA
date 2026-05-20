# Sistema de Prediccion de Ventas Rossmann
## Proyecto Final — Inteligencia Artificial 045
### Universidad Mariano Galvez de Guatemala

---

## Descripcion del Proyecto

Sistema de IA de extremo a extremo para **prediccion de demanda y gestion de inventario**
en el dominio Retail/Logistica, construido sobre el dataset publico
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)
(1,017,209 registros, 1,115 tiendas).

El sistema integra cuatro paradigmas de IA:
- **Modulo A:** Agente con busqueda/CSP para planificacion de inventario
- **Modulo B:** Pipeline ML supervisado (Random Forest vs Regresion Lineal)
- **Modulo C:** Red Neuronal Densa en PyTorch
- **Modulo D:** Componente NLP/LLM (clasificacion de demanda + resumen ejecutivo)

---

## Arquitectura del Sistema

```
[train.csv]
     |
     v
[preprocess.py] --> train_cleaned.csv (50k muestra)
     |
     +----> [train.py] ---------> modelo_rf.pkl, scaler.pkl
     |                            modelo_lr.pkl
     |
     +----> [train_dl.py] ------> mejor_modelo_dl.pth, history.json
     |
     v
[pipeline.py] <-- Modulo de Integracion (Modulo E)
     |
     +-- ML (Random Forest)    MAE: 1,697.69
     +-- DL (Red Neuronal)     MAE: 1,987.30
     +-- NLP (Groq LLM)        -> Clasificacion de demanda + Resumen ejecutivo
     |
     v
[Resultado Final]: Informe ejecutivo de prediccion de ventas
```

---

## Requisitos Previos

- Python 3.10+
- pip

**Dependencias de sistema (sin GPU requerida):**
```bash
pip install -r requirements.txt
```

**Configuración de la API Key de Groq (Opcional para NLP):**

Para habilitar el resumen ejecutivo avanzado generado por la API de Groq en lugar del fallback local:

1. Copia el archivo `.env.example` en la raíz del proyecto y renómbralo como `.env`:
   ```bash
   cp .env.example .env
   ```
2. Abre el archivo `.env` recién creado y coloca tu API key de Groq activa:
   ```env
   GROQ_API_KEY="tu_api_key_de_groq_aqui"
   ```

> [!TIP]
> Si no configuras la API key, el pipeline completo seguirá funcionando perfectamente en **modo fallback offline**, utilizando plantillas estructuradas deterministas locales. Puedes obtener una API key gratuita de Groq en [console.groq.com](https://console.groq.com).

---

## Instrucciones de Reproduccion (desde cero)

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-ia
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset

Descargar `train.csv` desde:
https://www.kaggle.com/c/rossmann-store-sales/data

Colocar el archivo en: `src/data/train.csv`

### 4. Ejecutar el preprocesamiento (Modulo B)

```bash
python src/ml/preprocess.py
```
Genera: `src/data/train_cleaned.csv` (50,000 registros limpios)

### 5. Entrenar modelos ML (Modulo B)

```bash
python src/ml/train.py
```
Genera: `src/ml/modelo_rf.pkl`, `src/ml/modelo_lr.pkl`, `src/ml/scaler.pkl`

### 6. Evaluar modelos ML

```bash
python src/ml/evaluate.py
```

### 7. Entrenar red neuronal (Módulo C)

```bash
python src/dl/train_dl.py
```
Genera: `src/dl/mejor_modelo_dl.pth`, `src/dl/history.json` y `src/dl/predictions_sample.npz`

> [!NOTE]
> **Archivos Pre-entrenados Disponibles:** Para facilitar una evaluación inmediata, el repositorio **ya incluye estos archivos pre-generados** (tanto los de Machine Learning en `src/ml/` como los de Deep Learning en `src/dl/`). Esto permite ejecutar directamente el notebook de análisis, el demo de búsqueda y el pipeline integrador (`src/integration/pipeline.py`) sin necesidad de esperar entrenamientos.
>
> **Consejo para el Equipo (Git):** Si vuelves a ejecutar los entrenamientos locales, los archivos se regenerarán con ligeras diferencias debido a inicializaciones estocásticas locales. Para evitar conflictos de merge o commits binarios accidentales, dile a Git que ignore tus cambios locales ejecutando:
> ```bash
> # Para Machine Learning
> git update-index --assume-unchanged src/ml/scaler.pkl src/ml/modelo_rf.pkl src/ml/modelo_lr.pkl
>
> # Para Deep Learning
> git update-index --assume-unchanged src/dl/mejor_modelo_dl.pth src/dl/history.json src/dl/predictions_sample.npz
> ```


### 8. Ejecutar el pipeline completo (Modulo E)

```bash
# Con API de Groq (resumen ejecutivo via LLM):
python src/integration/pipeline.py

# Sin API key (modo fallback):
python src/integration/pipeline.py --no-api
```

### 9. Ejecutar validación de sesgos y equidad (Módulo E)

```bash
python src/integration/verify_ethics.py
```
Genera un análisis en caliente detallando la disparidad de errores (MAE) por días con promoción y por ID de tienda para auditar el sistema.

### 10. Explorar los notebooks

Abrir en Jupyter o VSCode:
- `notebooks/ml_analysis.ipynb` — Analisis comparativo de modelos ML
- `notebooks/dl_analysis.ipynb` — Curvas de entrenamiento de la red neuronal
- `notebooks/nlp_demo.ipynb` — Comparativa TF-IDF vs Embeddings vs LLM

---

## Estructura del Proyecto

```
proyecto-ia/
├── src/
│   ├── data/
│   │   ├── load_data.py          # Carga y estadisticas del dataset
│   │   ├── train.csv             # Dataset original (NO incluido en repo, ver paso 3)
│   │   └── train_cleaned.csv     # Dataset limpio (generado por preprocess.py)
│   ├── search_csp/
│   │   ├── agent.py              # Agente de inventario (Modulo A)
│   │   └── algorithm.py         # Algoritmo CSP backtracking (Modulo A)
│   ├── ml/
│   │   ├── preprocess.py         # Limpieza y preprocesamiento
│   │   ├── train.py              # Entrenamiento LR y Random Forest
│   │   └── evaluate.py          # Evaluacion comparativa de modelos
│   ├── dl/
│   │   ├── model.py              # Arquitectura SalesDeepRegressor (PyTorch)
│   │   └── train_dl.py          # Bucle de entrenamiento DL
│   ├── nlp/
│   │   └── nlp_component.py     # TF-IDF, Embeddings, Groq LLM (Modulo D)
│   └── integration/
│       ├── pipeline.py          # Pipeline completo de extremo a extremo (Modulo E)
│       └── verify_ethics.py     # Auditoría de sesgos y equidad en vivo (Modulo E)
├── notebooks/
│   ├── demo_search.ipynb        # Demostracion del Agente CSP (Modulo A)
│   ├── ml_analysis.ipynb        # Visualizaciones ML
│   ├── dl_analysis.ipynb        # Curvas de entrenamiento DL
│   └── nlp_demo.ipynb           # Comparativa de enfoques NLP
├── docs/
│   ├── propuesta.md             # Propuesta inicial del proyecto
│   ├── asignacion_modulos.md    # Responsables por modulo
│   ├── agente_formulacion.md    # Formulacion del agente (Modulo A)
│   ├── arquitectura.md          # Diagrama de arquitectura de sistema
│   ├── search_decisions.md      # Decisiones de busqueda/CSP
│   ├── ml_decisions.md          # Decisiones de Machine Learning
│   ├── dl_decisions.md          # Decisiones de Deep Learning
│   ├── nlp_decisions.md         # Decisiones de NLP/LLM (comparativa cuantitativa)
│   └── ethics_analysis.md       # Analisis etico con evidencia real
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

---

## Resultados Clave

| Módulo | Modelo | Métrica | Valor |
|--------|--------|---------|-------|
| B — ML | Random Forest | MAE | 1,697.69 |
| B — ML | Regresión Lineal | MAE | 2,078.54 |
| C — DL | Red Neuronal (PyTorch) | MAE | 1,987.30 |
| D — NLP | TF-IDF + LR | Accuracy | ~0.72 |
| D — NLP | Embeddings + LR | Accuracy | ~0.79 |
| D — NLP | Groq LLM (zero-shot) | Accuracy | ~0.81 |

**Modelo ganador para predicción de ventas:** Random Forest (menor MAE, mayor robustez en datos tabulares)  
**Componente NLP elegido:** Groq LLM (único enfoque con capacidad generativa)

---

## Preguntas de Negocio Respondidas

1. **¿Cuánto venderá una tienda un día específico?** → Módulos B/C: predicción numérica con MAE ~1,697
2. **¿Es un día de Alta, Normal o Baja demanda?** → Módulo D: clasificación de contexto vía LLM
3. **¿Qué inventario debe ordenar la tienda?** → Módulo A: planificación CSP
4. **¿El sistema es justo para todas las tiendas?** → Módulo E: ratio de disparidad 4.68x detectado
---

## Integrantes y Modulos

Ver: [docs/asignacion_modulos.md](docs/asignacion_modulos.md)
