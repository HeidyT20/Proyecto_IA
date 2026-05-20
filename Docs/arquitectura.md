
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
                    │  Algoritmo: Backtracking con poda           │
                    │  Heurística: Evaluacion de costo            │
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
Ambos módulos procesan el dataset de forma paralela para generar predicciones independientes. El **Módulo B** utiliza Machine Learning supervisado clásico (Random Forest), mientras que el **Módulo C** emplea Deep Learning para series temporales (Red Neuronal en PyTorch). 

Para garantizar la máxima precisión de manera dinámica, el Módulo E de integración ejecuta una **selección automática en caliente** comparando el Error Medio Absoluto (MAE) en el conjunto de prueba. El sistema selecciona automáticamente el modelo con el menor error como su predicción primaria $P₀$. Actualmente, el modelo **Random Forest (ML)** es seleccionado por el pipeline al presentar un menor error (MAE ~1697.69 vs ~1987.30 del modelo de Deep Learning).

### Módulo A: Agente de Optimización (CSP)
Previo a la ejecución del agente, el sistema inyecta en memoria variables de negocio simuladas que no están en el dataset original (inventario inicial $I$, presupuesto $B$, órdenes pendientes $O$ y costos). Con esto y la predicción seleccionada $P₀$, el agente forma su estado inicial $S = \{I, t, P, O, B\}$. 

Resuelve el problema con **Backtracking con poda**, evaluando el costo total para encontrar las cantidades de pedido óptimas. La restricción del **nivel de servicio ≥ 95%** (evitar quiebres de stock) se modela de manera matemática rigurosa en la función de costo objetivo del CSP aplicando una penalización severa a la escasez: un factor de penalización de **10** para la escasez/quiebre de stock frente a un factor de tan solo **2** para el costo de almacenamiento. Esto obliga al agente a priorizar el abastecimiento y la satisfacción del cliente por encima del ahorro de almacenamiento.

### Módulo D: NLP / LLM
Traduce el plan matemático y de pedidos producido por el Módulo A a un reporte ejecutivo en lenguaje natural y en español. Explica de forma clara y justificada qué productos ordenar, en qué cantidad exacta absoluta y los fundamentos técnicos basados en los errores de los modelos de predicción. 

Para lograr esto con calidad de producción, el sistema utiliza el modelo de lenguaje de última generación **Llama 3.3 70B (`llama-3.3-70b-versatile`) a través de la API de Groq**. Este LLM ofrece un razonamiento avanzado para resúmenes de negocio con una latencia extremadamente baja y un cumplimiento estricto del formato sin alucinaciones de porcentajes. En caso de no contar con conexión o API key, se activa una degradación elegante mediante una plantilla estructurada determinista.

### Módulo E: Integración y Ética
Es el orquestador final del sistema. A través de `pipeline.py`, automatiza todo el flujo desde la carga de datos, la evaluación paralela de modelos, la selección del mejor predictor, la ejecución de la optimización CSP y la llamada al LLM para el reporte final. Además, evalúa éticamente los resultados, buscando posibles sesgos en las predicciones y centralizando toda la documentación global.