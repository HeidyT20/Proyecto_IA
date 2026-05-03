def copiar_estado(estado):
    return {
        "inventario": estado["inventario"].copy(),
        "demanda": estado["demanda"],
        "presupuesto": estado["presupuesto"]
    }


def es_valido(estado, producto, cantidad, costos, capacidad):
    # Restricción de presupuesto
    if estado["presupuesto"] < costos[producto] * cantidad:
        return False

    # Restricción de capacidad
    total = sum(estado["inventario"].values()) + cantidad
    if total > capacidad:
        return False

    return True


def backtracking(estado, productos, costos, capacidad, index=0):
    if index == len(productos):
        return estado

    producto = productos[index]

    mejor_resultado = None
    mejor_costo = float("inf")

    for cantidad in range(0, 51):  # dominio de valores

        if es_valido(estado, producto, cantidad, costos, capacidad):

            nuevo_estado = copiar_estado(estado)

            # aplicar acción
            nuevo_estado["inventario"][producto] += cantidad
            nuevo_estado["presupuesto"] -= costos[producto] * cantidad

            resultado = backtracking(
                nuevo_estado, productos, costos, capacidad, index + 1
            )

            if resultado:
                costo = evaluar(resultado)

                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_resultado = resultado

    return mejor_resultado


def evaluar(estado):
    costo = 0

    for p in estado["inventario"]:
        inventario = estado["inventario"][p]
        demanda = estado["demanda"][p]

        # almacenamiento
        costo += inventario * 2

        # escasez
        if inventario < demanda:
            costo += (demanda - inventario) * 10

    return costo