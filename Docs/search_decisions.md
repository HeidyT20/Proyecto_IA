# Decisión de Búsqueda y Diseño del CSP
## Optimización de Inventario con Backtracking en Python Puro
---

### 1. Contexto y Modelado del Problema
El problema de gestión de inventario se modela como un Problema de Satisfacción de Restricciones con Optimización (CSP-O). 
- **Variables**: Cantidad a reabastecer de cada producto (`A`, `B`, `C`).
- **Dominio**: Valores enteros discretos `[0, 50]` por producto.
- **Restricciones Globales**:
  - `Presupuesto`: `Σ(costo_p × cantidad_p) ≤ presupuesto_inicial`
  - `Capacidad`: `Σ(inventario_inicial_p + cantidad_p) ≤ capacidad_almacén`
- **Función Objetivo**: Minimizar el costo total (almacenamiento `×2` + penalización por déficit `×10`).

### 2. Algoritmo Seleccionado: Backtracking con Poda y Optimización
Se implementó un algoritmo de Backtracking recursivo con búsqueda exhaustiva y registro del mejor estado. 
- Explora sistemáticamente el árbol de decisiones asignando valores a cada variable.
- Aplica poda temprana mediante `es_valido()` antes de profundizar en ramas inviables.
- Al llegar a una hoja (todas las variables asignadas), evalúa el costo con `evaluar()` y mantiene el mínimo global (`mejor_costo`).

### 3. Justificación Técnica vs. Alternativas
| Algoritmo | ¿Por qué NO se seleccionó? |
|-----------|----------------------------|
| **BFS / DFS** | BFS consume memoria exponencial en profundidad. DFS sin backtracking no garantiza optimalidad ni maneja restricciones eficientemente. |
| **A\*** | Requiere una heurística admisible y consistente. Diseñar `h(n)` que subestime el costo real con restricciones globales (presupuesto/capacidad) es complejo y propenso a sobreoptimización. |
| **AC-3 / MRV** | Efectivos para CSP de *satisfacción* con restricciones binarias. Aquí las restricciones son n-arias/globales (suma de todos los productos), por lo que la propagación de arcos no aporta ventaja significativa frente a la verificación directa en `es_valido()`. |
| **Backtracking** | Ideal para espacios discretos acotados. Garantiza optimalidad global, es trivial de depurar, y se ajusta perfectamente a la arquitectura de poda por restricciones. |

### 4. Arquitectura del Módulo (`src/search_csp/`)
| Archivo | Responsabilidad |
|---------|-----------------|
| `agent.py` | Define `InventoryAgent`. Encapsula la lógica de dominio: validación de metas (`es_meta`) y cálculo de métricas (`costo_total`). Actúa como interfaz entre el algoritmo y la evaluación del negocio. |
| `algorithm.py` | Implementación pura del CSP. Contiene `backtracking()`, `es_valido()`, `copiar_estado()` y `evaluar()`. No depende de librerías externas; usa recursión y copias superficiales para garantizar inmutabilidad del estado padre. |
| `demo.ipynb` | Notebook demostrativo. Inicializa el agente, ejecuta el algoritmo sobre un caso realista y visualiza resultados, cumplimiento de demanda y análisis de costos. |

### 5. Espacio de Estados y Restricciones
- **Estado**: `{"inventario": {p: int}, "demanda": {p: int}, "presupuesto": int}`
- **Transiciones**: `nuevo_estado["inventario"][p] += cantidad` y `presupuesto -= costos[p] * cantidad`
- **Validación**: `es_valido()` verifica presupuesto y capacidad antes de asignar, evitando explorar ramas inviables.
- **Evaluación**: `evaluar()` y `agente.costo_total()` son funcionalmente equivalentes, separando lógica de búsqueda de lógica de negocio.

### 6. Complejidad y Escalabilidad
- **Complejidad Temporal**: `O(d^n)` donde `d = 51` (dominio) y `n = 3` (productos) → ~132,651 nodos máximos.
- **Complejidad Espacial**: `O(n)` por profundidad de pila recursiva.
- **Escalabilidad**: Para `n > 10` o dominios mayores, se recomendaría integrar **Forward Checking**, **Heurística LCV/MRV**, o migrar a solvers como `python-constraint`/`OR-Tools`. No obstante, para el alcance actual, el backtracking puro es óptimo en relación costo/beneficio.

### 7. Conclusión
La elección de Backtracking con poda y evaluación exhaustiva responde a los requisitos de corrección, transparencia y optimización garantizada. La implementación en Python puro cumple con la restricción de no usar librerías de IA de alto nivel, manteniendo el código legible, modular y fácilmente auditable. La separación entre `algorithm.py` (motor de búsqueda) y `agent.py` (lógica de dominio) sigue principios de ingeniería de software y facilita futuras extensiones (ej. integración con APIs de datos o heurísticas adicionales).

---
