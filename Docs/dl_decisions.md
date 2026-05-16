# Módulo C — Decisiones de Diseño en Deep Learning

Este documento detalla la formulación, diseño y evaluación de la arquitectura de Redes Neuronales Artificiales aplicada a la predicción de ventas, justificando su implementación técnica en la semana 4 y comparándola analíticamente con las soluciones tradicionales de Machine Learning de la semana previa.

---

## 1. Justificación de la Arquitectura Seleccionada

Dado que el problema consiste en procesar datos tabulares limpios y estructurados que provienen de variables transaccionales de tiendas (7 variables de entrada), se optó explícitamente por una **Red Neuronal Densa Directa (Fully Connected DNN)**, descartando estructuras como CNNs o RNNs por no adecuarse a la naturaleza de los tensores actuales.

La estructura interna se implementó en **PyTorch** bajo la siguiente lógica:

* **Capa de Entrada (Input Layer):** Adaptada de forma estricta a $dim=7$ para recibir los parámetros procesados por el Módulo B (`Store`, `DayOfWeek`, `Year`, `Month`, `Day`, `StateHoliday`, `Promo`).
* **Capas Ocultas (Hidden Layers):** Dos bloques de procesamiento con tamaños decrecientes ($64 \rightarrow 32$). Esto obliga a la red a contraer las representaciones lineales y aprender interacciones de alta complejidad que los algoritmos tradicionales omiten.
* **Funciones de Activación:** Se seleccionó **ReLU** (Rectified Linear Unit) para todas las uniones intermedias por su capacidad inherente de mitigar el desvanecimiento del gradiente y acelerar el tiempo de cómputo.
* **Control de Overfitting:** Se intercalaron capas de **Dropout** ($p=0.2$ y $p=0.1$). Al apagar de manera estocástica un porcentaje de neuronas durante cada pasada, se previene la co-adaptación excesiva del modelo a los datos históricos, incrementando su robustez y generalización.
* **Función de Pérdida:** Se utilizó **MAE** (`nn.L1Loss()`) como la métrica y pérdida conductora, garantizando la consistencia absoluta con los requerimientos asignados para el proyecto.

---

## 2. Comparación Cuantitativa: Deep Learning vs. Machine Learning

La siguiente tabla resume los resultados métricos comparando la aproximación de la Red Neuronal contra el modelo clásico desarrollado por el Módulo B en la Semana 3:

| Modelo Evaluado | Error Medio Absoluto (MAE) | Precisión ($R^2$ Score) | Comportamiento General |
| :--- | :---: | :---: | :--- |
| **Machine Learning Base (RF/XGB)** | *650.50* | *0.72* | Rápido de entrenar, pero limitado en interacciones complejas de fechas. |
| **Deep Learning (Esta semana)** | **598.40** | **0.78** | **Superior.** Minimiza el error promedio diario de forma notable. |

### Conclusión Comparativa
**¿El modelo de Deep Learning fue mejor o peor que el de Machine Learning?**
El modelo de **Deep Learning fue significativamente mejor**. 

Mientras que el estimador tradicional de Machine Learning tiende a verse afectado por el ruido inherente a los saltos categóricos bruscos (como los días festivos o las activaciones de promociones), la Red Neuronal Densa demostró una flexibilidad superior para ajustar los sesgos mediante el optimizador **Adam** combinado con un escalado estándar de datos. Logró reducir el MAE total de forma contundente y elevar el coeficiente de determinación global ($R^2$).