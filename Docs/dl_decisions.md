# Módulo C — Decisiones de Diseño en Deep Learning

Este documento detalla la formulación, diseño y evaluación de la arquitectura de Redes Neuronales Artificiales aplicada a la predicción de ventas, justificando su implementación técnica en la semana 4 y comparándola analíticamente con las soluciones tradicionales de Machine Learning de la semana previa.

---

## 1. Justificación de la Arquitectura Seleccionada

Dado que el problema consiste en procesar datos tabulares limpios y estructurados que provienen de variables transaccionales de tiendas (8 variables de entrada), se optó explícitamente por una *Red Neuronal Densa Directa (Fully Connected DNN)*, descartando estructuras como CNNs o RNNs por no adecuarse a la naturaleza de los tensores actuales.

La estructura interna se implementó en *PyTorch* bajo la siguiente lógica:

* *Capa de Entrada (Input Layer):* Adaptada de forma estricta a $dim=8$ para recibir los parámetros procesados por el Módulo B (Store, DayOfWeek, Year, Month, Day, StateHoliday, Promo, SchoolHoliday).
* *Capas Ocultas (Hidden Layers):* Dos bloques de procesamiento con tamaños decrecientes ($64 \rightarrow 32$). Esto obliga a la red a contraer las representaciones lineales y aprender interacciones de alta complejidad que los algoritmos tradicionales omiten.
* *Funciones de Activación:* Se seleccionó *ReLU* (Rectified Linear Unit) para todas las uniones intermedias por su capacidad inherente de mitigar el desvanecimiento del gradiente y acelerar el tiempo de cómputo.
* *Control de Overfitting:* Se intercalaron capas de *Dropout* ($p=0.2$ y $p=0.1$). Al apagar de manera estocástica un porcentaje de neuronas durante cada pasada, se previene la co-adaptación excesiva del modelo a los datos históricos, incrementando su robustez y generalización.
* *Función de Pérdida:* Se utilizó *MAE* (nn.L1Loss()) como la métrica y pérdida conductora, garantizando la consistencia absoluta con los requerimientos asignados para el proyecto.

---

## 2. Comparación Cuantitativa: Deep Learning vs. Machine Learning

La siguiente tabla resume los resultados métricos comparando la aproximación de la Red Neuronal contra el modelo clásico desarrollado por el Módulo B en la Semana 3:

| Modelo Evaluado | Error Medio Absoluto (MAE) | Precisión ($R^2$ Score) | Comportamiento General |
| :--- | :---: | :---: | :--- |
| *Machine Learning Base (Random Forest)* | ~1697.55 | ~0.61 | Rápido de entrenar, excelente particionando interacciones categóricas complejas de fechas. |
| *Deep Learning (Esta semana)* | *1987.30* | *~0.07* | *Inferior en este contexto.* El MAE es mayor y el R2 es sustancialmente menor. |

### Conclusión Comparativa
*¿El modelo de Deep Learning fue mejor o peor que el de Machine Learning?*
El modelo de *Deep Learning fue significativamente inferior* al Random Forest. 

Aunque las Redes Neuronales son excelentes para datos no estructurados (imágenes, texto), en los datos *tabulares* donde abundan variables categóricas complejas y discontinuidades lógicas (como tiendas específicas, días festivos), los modelos basados en árboles de decisión (como Random Forest o XGBoost) suelen superarlas sin requerir de complejas técnicas de "Feature Embedding". La Red Neuronal Densa implementada presentó mayores dificultades para igualar el bajo error diario (MAE) que el Random Forest logró alcanzar.