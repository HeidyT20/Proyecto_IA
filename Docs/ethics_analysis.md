# Analisis etico (Modulo E)

## Alcance
Este analisis se enfoca en fairness y sesgo para la prediccion de demanda y decisiones de inventario. La unidad principal de analisis es Store (Store ID) del dataset Rossmann.

## Criterios de fairness
- Agrupacion: Store (Store ID)
- Metrica: mean absolute error (MAE) por tienda
- Objetivo: reducir brechas grandes de error entre tiendas

## Metodo
1. Entrenar el modelo (Modulo B/C).
2. Predecir demanda en el conjunto de validacion.
3. Calcular MAE por tienda.
4. Comparar las 10 tiendas con mayor error vs. el MAE global.
5. Reportar disparidades y proponer mitigaciones.

## Evidencia (placeholder)
- MAE global: TODO
- MAE top 10 tiendas: TODO
- Ratio de disparidad (Top10 / Global): TODO

## Riesgos
- Tiendas con bajo trafico pueden estar subrepresentadas y mostrar mayor error.
- Periodos con promo pueden sesgar la prediccion y causar sobre-orden.

## Mitigaciones
- Rebalancear muestras por tienda o agregar features a nivel tienda.
- Evaluar un modelo robusto (ej. Gradient Boosting) vs. baseline.
- Aplicar reglas de post-procesamiento para tiendas de alto riesgo (revision manual).

## Limitaciones
- El fairness depende de la calidad y cobertura de datos.
- Es un analisis a nivel de grupo; no prueba causalidad.
