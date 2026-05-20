import torch
import torch.nn as nn

class SalesDeepRegressor(nn.Module):
    def __init__(self, input_dim=8):
        super(SalesDeepRegressor, self).__init__()
        
        # Definición de la arquitectura secuencial de la Red Densa
        self.network = nn.Sequential(
            # Capa de entrada (8 features) -> Primera capa oculta (64 neuronas)
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Reduce el sobreajuste apagando el 20% de las neuronas aleatoriamente
            
            # Segunda capa oculta (64 neuronas) -> Tercera capa oculta (32 neuronas)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            # Tercera capa oculta (32 neuronas) -> Capa de salida (1 neurona para predicción de valor continuo)
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        Define el flujo hacia adelante de los datos (Forward Pass).
        """
        return self.network(x)

if __name__ == "__main__":
    # Prueba rápida de la arquitectura con un tensor de ejemplo
    model = SalesDeepRegressor(input_dim=8)
    print("--- Estructura del Modelo de Red Neuronal (PyTorch) ---")
    print(model)
    
    # Simular una entrada de 2 registros y 8 variables
    test_input = torch.randn(2, 8)
    test_output = model(test_input)
    print(f"\nForma del tensor de salida de prueba: {test_output.shape} (Debe ser [2, 1])")