import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    print("Iniciando preprocesamiento...")
    # Cargar datos (usamos una muestra si es muy grande o el archivo completo)
    df = pd.read_csv(input_path, low_memory=False)
    
    # 1. Manejo de fechas
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # 2. Convertir variables categóricas a números
    df['StateHoliday'] = df['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3, 0:0})
    
    # 3. Eliminar filas donde las tiendas estaban cerradas (ventas = 0)
    df = df[df['Open'] != 0]
    
    # 4. Seleccionar columnas finales para el modelo
    features = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'Promo']
    target = 'Sales'
    
    df_final = df[features + [target]]
    
    # Guardar los datos limpios para el siguiente paso
    df_final.to_csv(output_path, index=False)
    print(f"Datos limpios guardados en: {output_path}")

if __name__ == "__main__":
    preprocess_data("src/data/train.csv", "src/data/train_cleaned.csv")