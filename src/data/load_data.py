import pandas as pd
import os

def cargar_y_explorar_datos():
    # Nombre del archivo que ya tienes en la carpeta
    archivo = "train.csv"
    
    # Verificar si el archivo existe en la ruta actual
    if not os.path.exists(archivo):
        print(f"Error: No se encontró el archivo '{archivo}' en la carpeta actual.")
        print("Asegúrate de que el archivo .csv esté en: src/data/")
        return

    print("Cargando dataset... (esto puede tardar unos segundos debido al tamaño)")
    
    try:
        # 1. Cargar el dataset
        # low_memory=False se usa para evitar advertencias por el tamaño del archivo
        df = pd.read_csv(archivo, low_memory=False)

        print("\n" + "="*30)
        print("      EXPLORACIÓN DE DATOS")
        print("="*30)

        # 2. Mostrar Tamaño
        print(f"\n1. TAMAÑO DEL DATASET:")
        print(f"   - Total de registros (filas): {df.shape[0]}")
        print(f"   - Total de columnas: {df.shape[1]}")

        # 3. Mostrar Columnas y Tipos de Datos
        print("\n2. ESTRUCTURA Y TIPOS DE DATOS:")
        print(df.info())

        # 4. Mostrar Valores Nulos
        print("\n3. VALORES NULOS POR COLUMNA:")
        nulos = df.isnull().sum()
        print(nulos[nulos > 0] if nulos.sum() > 0 else "No se encontraron valores nulos.")

        # 5. Estadísticas Básicas
        print("\n4. ESTADÍSTICAS DESCRIPTIVAS:")
        print(df.describe())

        print("\n" + "="*30)
        print("   PROCESO COMPLETADO CON ÉXITO")
        print("="*30)

    except Exception as e:
        print(f"Ocurrió un error al procesar los datos: {e}")

if __name__ == "__main__":
    cargar_y_explorar_datos()