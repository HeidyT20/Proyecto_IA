# Fase 1: Propuesta del Proyecto de IA - Semana 1

## 1. Información General
* **Universidad:** Universidad Mariano Gálvez de Guatemala
* **Facultad:** Ingeniería en Sistemas de Información
* **Curso:** Inteligencia Artificial
* **Dominio Seleccionado:** Retail / Logística
* **Problema Concreto:** Predicción de demanda y optimización de niveles de inventario.

## 2. Descripción del Problema
En el sector retail, uno de los mayores desafíos es el equilibrio de inventario. Tener demasiado stock genera costos de almacenamiento y riesgo de merma, mientras que tener poco stock (quiebre de inventario) resulta en ventas perdidas y clientes insatisfechos. 

Este proyecto busca resolver este dilema mediante un sistema inteligente que:
1.  Prediga con precisión las ventas diarias futuras basándose en datos históricos, promociones y estacionalidad.
2.  Tome decisiones automatizadas sobre cuándo y cuánto producto pedir para mantener un nivel de stock óptimo.

## 3. Justificación del Dataset
Para el desarrollo de este proyecto se ha seleccionado el dataset real y público **Rossmann Store Sales**.

* **Fuente:** [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
* **Volumen de datos:** El dataset cuenta con **1,017,209 registros** en su conjunto de entrenamiento, superando ampliamente el requisito mínimo de 50,000 registros.
* **Relevancia:** Los datos incluyen variables críticas como ventas diarias, número de clientes, días festivos, promociones escolares y tipos de surtido de tienda, lo que permite un análisis profundo de patrones de consumo aplicables a farmacias y tiendas de conveniencia en contextos similares al de Guatemala.

### Columnas Clave a Utilizar:
| Columna | Descripción | Uso en el Proyecto |
| :--- | :--- | :--- |
| `Sales` | Cantidad de ventas diarias | Variable objetivo para ML y DL. |
| `Customers` | Número de clientes | Feature para entender la afluencia. |
| `Promo` | Indica si hay promoción activa | Factor determinante en el pico de demanda. |
| `StateHoliday` | Días festivos | Modelado de estacionalidad. |
| `Store` | ID de la sucursal | Permite la optimización por ubicación. |

## 4. Justificación del uso de IA (Paradigmas)
El proyecto integra tres componentes de IA según lo requerido:

1.  **Razonamiento Clásico (Módulo A):** Un agente inteligente utilizará algoritmos de búsqueda o CSP para determinar la política de reabastecimiento (ej. Punto de Reorden) basándose en las predicciones obtenidas.
2.  **Aprendizaje Supervisado (Módulo B):** Implementación de un pipeline de Machine Learning (ej. Random Forest o XGBoost) para la ingeniería de variables y predicción base.
3.  **Deep Learning (Módulo C):** Uso de Redes Neuronales Recurrentes (LSTM) para capturar dependencias temporales de largo plazo en las series de tiempo de ventas.
4.  **NLP (Módulo D):** Análisis de sentimiento o tendencias de mercado en reportes para ajustar los factores de riesgo en la cadena de suministro.

## 5. Impacto Esperado
Se espera que el sistema reduzca el error de predicción de demanda en comparación con métodos estadísticos tradicionales, permitiendo una gestión de logística "justo a tiempo" (Just-in-Time) que mejore la rentabilidad operativa de la empresa.
