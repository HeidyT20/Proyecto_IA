
---

# Diagrama de Arquitectura — Sistema IA: Retail / Logística
## Predicción de demanda + Optimización de inventario

---

```
                    ┌─────────────────────────────────────────────┐
                    │       DATASET — Rossmann Store Sales        │
                    │   Historial de ventas (1,115 tiendas)       │
                    │   (1,017,209 registros reales)              │
                    └──────────────────┬──────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                          ▼                         ▼
          ┌───────────────────────────┐ ┌───────────────────────────┐
          │       MÓDULO B            │ │       MÓDULO C            │
          │       Pipeline ML         │ │       Deep Learning       │
          │                           │ │                           │
          │  Predicción de demanda    │ │  Predicción de demanda    │
          │  (ML supervisado clásico) │ │  (Series Temporales)      │
          └─────────────┬─────────────┘ └─────────────┬─────────────┘
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                           P₀: demanda pronosticada 
                       + Variables de entorno simuladas 
                           (I, B, O, Costos, etc.)
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │              MÓDULO A                       │
                    │            Agente / CSP                     │
                    │                                             │
                    │  Optimización de pedidos:                   │
                    │  Estado: S = {I, t, P, O, B}                │
                    │  Minimizar: C_pedido + C_almacen + C_escasez│
                    │                                             │
                    │  Algoritmo: Backtracking + AC-3             │
                    │  Heurística: h(n) Admisible                 │
                    │  Restricciones: B, Capacidad, Servicio ≥ 95%│
                    └──────────────────┬──────────────────────────┘
                                       │
                              plan óptimo de pedidos
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │              MÓDULO D                       │
                    │             NLP / LLM                       │
                    │                                             │
                    │  Generación de resúmenes en lenguaje        │
                    │  natural del plan de pedidos óptimo         │
                    │  (qué ordenar, por qué, cuándo)             │
                    └──────────────────┬──────────────────────────┘
                                       │
                          reporte interpretable + contexto
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │              MÓDULO E                       │
                    │         Integración + Ética                 │
                    │                                             │
                    │  pipeline.py:                               │
                    │  datos → ML/DL → Agente → NLP → final       │
                    │                                             │
                    │  Análisis de sesgos y fairness              │
                    │  README + documentación global              │
                    └──────────────────┬──────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │           RESULTADO FINAL                   │
                    │                                             │
                    │  Plan de órdenes: qué pedir, cuánto y cuándo│
                    │  Costo total minimizado (Servicio ≥ 95%)    │
                    │  Reporte técnico y evaluación ética         │
                    └─────────────────────────────────────────────┘
```

---

## Explicación del sistema

### Punto de entrada: Dataset Rossmann
El sistema utiliza el dataset **Rossmann Store Sales**, el cual cuenta con más de **1 millón de registros**. Esta es la fuente de verdad primaria que contiene el historial de ventas reales. 

### Módulos B y C: Predicción de Demanda
Ambos módulos procesan el dataset de forma paralela para generar la predicción de demanda. El **Módulo B** utiliza Machine Learning supervisado clásico, mientras que el **Módulo C** emplea Deep Learning para series temporales. Sus salidas convergen en la predicción $P₀$.

### Módulo A: Agente de Optimización (CSP)
Previo a la ejecución del agente, el sistema inyecta en memoria variables de negocio simuladas que no están en el dataset original (inventario inicial $I$, presupuesto $B$, órdenes pendientes $O$ y costos). Con esto y la predicción $P₀$, el agente forma su estado inicial $S = \{I, t, P, O, B\}$. Resuelve el problema utilizando **Backtracking con AC-3** y una **heurística admisible** para encontrar las cantidades de pedido óptimas, minimizando costos (almacenamiento, escasez, pedido) y garantizando un nivel de servicio del 95%.

### Módulo D: NLP / LLM
Traduce el plan matemático producido por el Módulo A a un reporte en lenguaje natural, explicando al operador humano qué productos ordenar, en qué cantidad y la justificación técnica de la decisión basándose en las predicciones.

### Módulo E: Integración y Ética
Es el orquestador final del sistema. A través de `pipeline.py`, automatiza todo el flujo desde la carga de datos hasta el reporte final. Además, evalúa éticamente los resultados, buscando posibles sesgos en las predicciones (por ejemplo, si el modelo favorece injustamente a ciertas tiendas o productos) y centraliza la documentación.