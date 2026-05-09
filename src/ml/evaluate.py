import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate():
    df = pd.read_csv("src/data/train_cleaned.csv").head(1000) # Probar con una muestra
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    with open("src/ml/modelo_ventas.pkl", 'rb') as f:
        model = pickle.load(f)
    
    predicciones = model.predict(X)
    mae = mean_absolute_error(y, predicciones)
    r2 = r2_score(y, predicciones)
    
    print(f"Error Medio Absoluto: {mae:.2f}")
    print(f"Precisión (R2 Score): {r2:.2f}")

if __name__ == "__main__":
    evaluate()