import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Asegurar que el modulo local se puede importar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..', '..')))
from src.dl.model import SalesDeepRegressor

def train_deep_learning():
    print("Iniciando el entrenamiento real de Deep Learning con PyTorch...")
    
    # Rutas
    data_path = "src/data/train_cleaned.csv"
    scaler_path = "src/ml/scaler.pkl"
    base_salida = "src/dl"
    os.makedirs(base_salida, exist_ok=True)
    
    # 1. Cargar datos
    print("Cargando dataset limpio...")
    df = pd.read_csv(data_path)
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    # Mismo split que en Machine Learning para ser justos en la comparacion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Cargar escalador de la semana 3 y transformar
    print("Escalando datos...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Convertir a Tensores de PyTorch
    print("Convirtiendo a Tensores...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    
    # 4. Crear DataLoaders
    batch_size = 128
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Instanciar Modelo, Optimizador y Función de Pérdida
    model = SalesDeepRegressor(input_dim=8)
    criterion = nn.L1Loss() # Error Medio Absoluto (MAE)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Learning rate ajustado para convergencia rapida
    
    epochs = 40
    train_loss_history = []
    val_loss_history = []
    
    print(f"\nIniciando entrenamiento por {epochs} épocas...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Fase de entrenamiento
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                 outputs = model(batch_X)
                 loss = criterion(outputs, batch_y)
                 val_loss += loss.item() * batch_X.size(0)
                 
        val_loss /= len(val_loader.dataset)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{base_salida}/mejor_modelo_dl.pth")
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Época [{epoch+1:02d}/{epochs}] -> Loss: {train_loss:.2f} | Val_Loss: {val_loss:.2f}")
            
    print(f"\nEntrenamiento finalizado. Mejor Val_Loss (MAE): {best_val_loss:.2f}")
    
    # 6. Guardar historial
    history = {"train_loss": train_loss_history, "val_loss": val_loss_history}
    with open(f"{base_salida}/history.json", "w") as f:
        json.dump(history, f)
        
    # 7. Generar muestra de predicciones para el Notebook
    # Usamos los mejores pesos
    model.load_state_dict(torch.load(f"{base_salida}/mejor_modelo_dl.pth"))
    model.eval()
    with torch.no_grad():
        sample_X = X_val_tensor[:100]
        sample_y = y_val_tensor[:100]
        preds = model(sample_X)
        np.savez(f"{base_salida}/predictions_sample.npz", real=sample_y.numpy(), pred=preds.numpy())
        
    print(f"Archivos generados en {base_salida} (history.json, mejor_modelo_dl.pth, predictions_sample.npz)")

if _name_ == "_main_":
    train_deep_learning()