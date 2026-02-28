# %% ========================================================================
# Imports

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = 'cpu'


# %% ========================================================================
# Prepare data

Fun = lambda x: 100 * (np.sin(x[:,0] * x[:,1]) + np.cos(x[:,1] + x[:,0]))

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_outputs = 1 # liczba wyjść

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

X_min = 0
X_max = 10

X_train = np.random.uniform(X_min, X_max, (n_train, n_inputs))
Y_train = Fun(X_train).reshape(n_train, n_outputs)

Y_min = Y_train.min() # minimalna wartość Y w zbiorze uczącym
Y_max = Y_train.max() # maksymalna wartość Y w zbiorze uczącym

X_train = (X_train - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_train = (Y_train - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

X_val = np.random.uniform(X_min, X_max, (n_val, n_inputs))
Y_val = Fun(X_val).reshape(n_val, n_outputs)

X_val = (X_val - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_val = (Y_val - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

# ---

dataset = TensorDataset(
    torch.from_numpy(X_train.astype(np.float32)), 
    torch.from_numpy(Y_train.astype(np.float32)))

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

train_inputs = torch.from_numpy(X_train.astype(np.float32)).to(device)
train_targets = torch.from_numpy(Y_train.astype(np.float32)).to(device)
train_targets = train_targets.view(-1, 1) # Upewnienie się,

val_inputs = torch.from_numpy(X_val.astype(np.float32)).to(device)
val_targets = torch.from_numpy(Y_val.astype(np.float32)).to(device)
val_targets = val_targets.view(-1, 1) # Upewnienie się, że wymiary się zgadzają



# %% ========================================================================
# Define model 

class MLPmodel(nn.Module):
    def __init__(self, input_size, hidden_layers_size, hidden_layers_count, output_size):
        super(MLPmodel, self).__init__()
        
        self.input_layer = (nn.Linear(input_size, hidden_layers_size))
        
        self.middle_layers = nn.ModuleList([
            nn.Linear(hidden_layers_size, hidden_layers_size) 
            for _ in range(hidden_layers_count - 1)
        ])

        self.output_layer = nn.Linear(hidden_layers_size, output_size)
        
        self.activation = nn.GELU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.middle_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x
    
model = MLPmodel(n_inputs, 50, 8, n_outputs)

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())



# %% ========================================================================
# Train model

epochs = 30

for epoch in range(1, epochs + 1):
    model.train() # Ustawienie modelu w tryb treningu
    
    for i, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.view(-1, 1) # Upewnienie się, że wymiary się zgadzają

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        model.eval() # Ustawienie modelu w tryb ewaluacji
        with torch.no_grad():
            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs, train_targets)

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)

        print(f'Epoch {epoch}/{epochs}, Training Loss: {train_loss.item():.6e}, Validation Loss: {val_loss.item():.6e}')


model.eval() # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():
    train_outputs = model(train_inputs)
    train_loss = criterion(train_outputs, train_targets)

    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_targets)

print(f'Final Training Loss: {train_loss.item():.6e}, Final Validation Loss: {val_loss.item():.6e}')

Y_train_pred = train_outputs.cpu().numpy()
Y_val_pred = val_outputs.cpu().numpy()

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.scatter(Y_train, Y_train_pred, s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Train set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.scatter(Y_val, Y_val_pred, s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Val set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()


