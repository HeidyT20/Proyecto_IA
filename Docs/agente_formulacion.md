# Formulación de agentes — Sistema IA 

## Retail / Logística (Predicción de demanda + inventario)

---

### 1. Descripción del problema

El sistema debe determinar las cantidades óptimas de pedido para múltiples productos en un almacén, minimizando costos totales (almacenamiento, escasez y pedido) mientras se satisface la demanda pronosticada.


---
### 2. Formulación del Agente
#### ESTADO
Un estado se define como una tupla y representa la situación del sistema en un momento dado:

1. S = {I,t,P,O,B}

Donde:
- I: El nivel de inventaio actual para cada producto n
```I = {i₁, i₂, …, iₙ}```
- t: Período de tiempo actual (dia/semana)
- P: Predicción de demanda para los próximos k períodos
```P= {d₁, d₂, ..., dₖ}```
- O: Pedidos pendientes con sus fechas de llegada
```O = {o₁, o₂, ..., oₘ}```
- B: Presupuesto restante disponible para realizar compras

Ejemplo:

```
estado = {
    'inventario': {'producto_A': 150, 'producto_B': 75, 'producto_C': 200},
    'tiempo': 'semana_15',
    'demanda_predicha': {'producto_A': 120, 'producto_B': 90, 'producto_C': 180},
    'pedidos_pendientes': [{'producto': 'producto_A', 'cantidad': 100, 'llega': 'semana_16'}],
    'presupuesto_restante': 50000
}
```

___

#### ESTADO INICIAL

```
S₀ = {
    I₀: Niveles actuales de inventario (datos reales del dataset),
    t₀: Semana/día actual del período de planificación,
    P₀: Demanda pronosticada por el módulo de ML,
    O₀: Pedidos en tránsito al inicio,
    presupuesto: Presupuesto disponible para compras
}
```
---
#### ESTADO META (OBJETIVO)
Un estado es meta si cumple TODAS las siguientes condiciones:
1. Satisfacción de demanda:
   Para todo producto i: 
    ```inventario[i] + pedidos_pendientes[i] ≥ demanda_predicha[i] × factor_seguridad    ```
2. Restricción de presupuesto:
    ```costo_total_pedidos ≤ presupuesto_disponible     ```
3. Capacidad del almacén:
```Σ(inventario[i] × espacio_unitario[i]) ≤ capacidad_total_almacen```
4. Nivel de servicio mínimo:
```tasa_servicio ≥ 95% (o threshold definido)```
5. Optimización de costos:
```costo_total = costo_pedido + costo_almacenamiento + costo_escasez ```  sea mínimo
---
#### ACCIONES
Cada acción es una tupla: ```A = {producto, cantidad, tipo_acción}```
Acciones disponibles:

| Acción              | Descripción         | Predicciones       |
|--------------------------|--------------------------------------------|------------------------------------------------|
| ORDENAR (p, q)       | Pedir q unidades del producto p          | `costo(p,q) ≤ B` (presupuesto suficiente), `q ≥ 0`                 |
| ESPERAR()| No realizar pedidos en este período          | Ninguna  |
| TRANSFERIR(p,  q, almacén_origen, almacén_destino)   | Mover stock entre almacenes   | ```inventario[origen] ≥ q```    |
| AJUSTAR_DEMANDA(p, factor) | Ajustar predicción para factores externos     | Justificación documentada           |

Espacio de acciones:
- Para n productos y m cantidades posibles: O(n × m) acciones por estado
- Ejemplo: 100 productos × 50 niveles de cantidad = 5,000 acciones posibles
---
#### FUNCIÓN DE EVALUACIÓN (HEURÍSTICA)
Función de costo total a minimizar:
```
def funcion_evaluacion(estado):
    costo_pedido = Σ(costo_fijo_pedido + costo_unitario[p] × cantidad_pedida[p])
    
    costo_almacenamiento = Σ(inventario[p] × costo_almacenamiento_unitario[p])
    
    # Penalización por escasez (stockout)
    deficit = max(0, demanda_predicha[p] - (inventario[p] + pedidos_pendientes[p]))
    costo_escasez = Σ(deficit[p] × costo_stockout[p])
    
    # Penalización por exceso de inventario
    exceso = max(0, inventario[p] - demanda_maxima_esperada[p])
    costo_exceso = Σ(exceso[p] × tasa_obsolescencia[p])
    
    # Función objetivo
    costo_total = costo_pedido + costo_almacenamiento + costo_escasez + costo_exceso
    
    return costo_total
```

Heurística propuesta (para A o búsqueda informada)

```
def heuristica(estado_actual, estado_meta):
    """
    h(n) = Costo estimado mínimo para alcanzar el estado meta
    Heurística admisible: subestima el costo real
    """
    h = 0
    
    # Para cada producto con déficit proyectado
    for producto in productos:
        deficit = max(0, demanda_predicha[producto] - inventario_disponible[producto])
        
        # Costo mínimo estimado: pedir lo justo al menor costo unitario
        h += deficit × costo_unitario_minimo[producto]
        
        # Agregar penalización por tiempo de entrega si aplica
        if tiempo_entrega[producto] > 0:
            h += deficit × costo_espera_diario[producto] × tiempo_entrega[producto]
    
    return h

```

Justificación de la heurística:
- **Admisibilidad**: La heurística asume que se puede pedir la cantidad exacta necesaria sin costos fijos adicionales y sin restricciones de capacidad, lo cual subestima el costo real.
- **Consistencia**: Para cualquier estado n y sucesor n': ```h(n) ≤ costo(n, n') + h(n')```
- **Información**: Considera tanto el costo de adquisición como el costo temporal, proporcionando una guía más precisa que una heurística de "cero" o "número de productos con déficit".

___
### 3. Complejidad del Espacio de Búsqueda
- Factor de ramificación (b): Depende del número de productos (n) y cantidades de pedido posibles (q).
        ```b ≈ n × q ``` (ej: 100 productos × 50 cantidades = 5,000)
- Profundidad (d): Número de períodos de planificación (semanas/días)
        ```d = 4-12 ``` (para planificación mensual/trimestral)
- Tamaño del espacio: ``` O(b^d) = O((n×q)^d) ```
Conclusión: Espacio exponencial → requiere algoritmos informados (A*, branch-and-bound) o técnicas de optimización con restricciones (CSP)

---
### 4. Algoritmo Propuesto
Opción A: CSP (Constraint Satisfaction Problem)
- Variables: Cantidad a pedir para cada producto en cada período
- Dominios: {0, 1, 2, ..., max_pedido}
- Restricciones:
    - Hard: Presupuesto, capacidad, demanda mínima
    - Soft: Minimizar costos, maximizar servicio
- Algoritmo: Backtracking con AC-3 (Arc Consistency)

Opción B: Búsqueda A*
- Estado: Definido arriba
- Función de evaluación: f(n) = g(n) + h(n)
    - g(n): Costo acumulado desde el inicio
    - h(n): Heurística definida arriba
- Ventaja: Encuentra solución óptima si h(n) es admisible

Decisión final: Se implementará CSP porque:
1. El problema tiene muchas restricciones hard (presupuesto, capacidad)
2. Las variables son discretas (cantidades enteras)
3. CSP permite propagación de restricciones y poda temprana
4. Más eficiente que búsqueda ciega en espacios con muchas restricciones
---
### 5. Ejemplo de Caso del Dominio
Escenario: Almacén de retail con 3 productos, planificación semanal
```
Estado inicial:
- Inventario: {A: 50, B: 30, C: 100}
- Demanda predicha (semana 16): {A: 80, B: 40, C: 90}
- Presupuesto: $10,000
- Costos: {A: $50/unid, B: $100/unid, C: $30/unid}

Acción óptima:
- ORDENAR(A, 30) → Costo: $1,500
- ORDENAR(B, 10) → Costo: $1,000
- No ordenar C (inventario suficiente)

Estado resultante:
- Inventario proyectado: {A: 80, B: 40, C: 100}
- Demanda satisfecha: 100%
- Costo total: $2,500
- Presupuesto restante: $7,500

```
---
## 6. Integración con Módulo B y Dataset Rossmann

**Fuente de datos para el agente:**

- **Entrada P (Predicción de demanda):** Proviene del Módulo B (ML/DL) que procesará el dataset Rossmann Store Sales (1,017,209 registros). El Módulo B entregará `ventas_predichas_{tienda}_{día}` como insumo principal del estado del agente.

**Variables de simulación (no presentes en el dataset original):**
El dataset Rossmann contiene ventas históricas pero NO incluye inventario físico, costos operativos ni presupuesto. Para mantener el enfoque en el algoritmo de decisión (CSP), se definirán parámetros de negocio realistas:

| Variable | Fuente/Justificación |
|----------|---------------------|
| `inventario_inicial (I)` | Simulado proporcionalmente a `Sales` promedio por tienda |
| `presupuesto_restante (B)` | Calculado como: `ventas_históricas_promedio × margen_estimado_30%` |
| `costo_unitario` | Fijo por categoría de producto (simulación controlada) |
| `capacidad_almacen` | Proporcional al tamaño de tienda (`Store` + `Assortment` del dataset) |
| `costo_almacenamiento` | 2-5% del valor del inventario (estándar retail) |
| `penalización_stockout` | 3× el margen de ganancia (pérdida de venta + insatisfacción) |

