import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model(data_path, model_path):
    print("Entrenando el modelo...")
    df = pd.read_csv(data_path)
    
    X = df.drop('Sales', axis=1) # Características
    y = df['Sales']             # Lo que queremos predecir
    
    # Crear y entrenar el modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Guardar el modelo entrenado en un archivo
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {model_path}")

if __name__ == "__main__":
    train_model("src/data/train_cleaned.csv", "src/ml/modelo_ventas.pkl")