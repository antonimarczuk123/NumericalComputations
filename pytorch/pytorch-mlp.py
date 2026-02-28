# %% ========================================================================
# Imports

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# %% ========================================================================
# Prepare data

Fun = lambda x: 100 * (np.sin(x[:,0] * x[:,1]) + np.cos(x[:,1] + x[:,0]))

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_outputs = 1 # liczba wyjść

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

X_min = 0
X_max = 10

X_train = np.random.uniform(X_min, X_max, (n_train, n_inputs)).astype(np.float32)
Y_train = Fun(X_train)

dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



# %% ========================================================================
# Define model 

class MyDeepNet(nn.Module):
    def __init__(self, input_size, hidden_layers_size, hidden_layers_count, output_size):
        super(MyDeepNet, self).__init__()
        
        # Tworzymy listę warstw (ModuleList pozwala PyTorchowi "widzieć" te warstwy)
        self.layers = nn.ModuleList()
        
        # Pierwsza warstwa
        self.layers.append(nn.Linear(input_size, hidden_layers_size))
        
        # Warstwy ukryte
        for i in range(hidden_layers_count - 1):
            self.layers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            
        # Warstwa wyjściowa
        self.output_layer = nn.Linear(hidden_layers_size, output_size)
        
        # Funkcja aktywacji
        self.activation = nn.GELU()
        
        # Wywołujemy inicjalizację wag
        self._init_weights()

    def _init_weights(self):
        """Prywatna metoda do inicjalizacji wag"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Definicja przepływu danych (Forward Pass)"""
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Na wyjściu zazwyczaj nie dajemy aktywacji w regresji
        x = self.output_layer(x)
        return x
    

hidden_config = [50 for _ in range(8)]
model = MyDeepNet(n_inputs, hidden_config, n_outputs)

device = 'cpu'
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# %% ========================================================================
# Train model

epochs = 20

for epoch in range(epochs):
    model.train() # Ustawienie modelu w tryb treningu
    running_loss = 0.0
    
    for i, (inputs, targets) in enumerate(dataloader):
        # 1. Przeniesienie danych na urządzenie
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.view(-1, 1) # Upewnienie się, że wymiary się zgadzają

        # 2. Wyzerowanie gradientów (bardzo ważne!)
        optimizer.zero_grad()

        # 3. Forward pass (obliczenia modelu)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 4. Backward pass (obliczanie pochodnych)
        loss.backward()

        # 5. Aktualizacja wag
        optimizer.step()

        running_loss += loss.item()
    
    # Logowanie postępu
    if (epoch + 1) % 10 == 0:
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")




