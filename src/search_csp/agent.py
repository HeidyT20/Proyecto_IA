class InventoryAgent:
    def __init__(self, productos, costos, capacidad_almacen):
        self.productos = productos
        self.costos = costos
        self.capacidad = capacidad_almacen

    def es_meta(self, estado):
        # 1. Satisfacción de demanda
        for p in self.productos:
            if estado["inventario"][p] < estado["demanda"][p]:
                return False

        # 2. Restricción de presupuesto
        if estado["presupuesto"] < 0:
            return False

        # 3. Capacidad de almacén (simplificada)
        total = sum(estado["inventario"].values())
        if total > self.capacidad:
            return False

        return True

    def costo_total(self, estado):
        costo = 0

        for p in self.productos:
            inventario = estado["inventario"][p]
            demanda = estado["demanda"][p]

            # costo almacenamiento
            costo += inventario * 2  

            # costo escasez
            if inventario < demanda:
                deficit = demanda - inventario
                costo += deficit * 10  

        return costo