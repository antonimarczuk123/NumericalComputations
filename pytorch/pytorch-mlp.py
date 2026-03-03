# %% ========================================================================
# Imports

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = 'cpu'


# %% ========================================================================
# Prepare data

Fun = lambda x: 100 * (np.sin(x[:,0] * x[:,1]) + np.cos(x[:,1] + x[:,0]))

n_inputs = 2
n_outputs = 1

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

train_dataset = TensorDataset(
    torch.from_numpy(X_train.astype(np.float32)), 
    torch.from_numpy(Y_train.astype(np.float32))
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(
    torch.from_numpy(X_val.astype(np.float32)), 
    torch.from_numpy(Y_val.astype(np.float32))
)

val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# %% ========================================================================
# Define model and initialize weights

class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()

        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc5(x)
        return x
    
model = MLPmodel().to(device)
    

# %% ========================================================================
# Training

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50

train_losses = np.zeros(n_epochs)
val_losses = np.zeros(n_epochs)

for epoch in range(n_epochs):
    model.train()
    running_train_loss = 0.0

    for X_batch, Y_batch in train_dataloader:
        optimizer.zero_grad()

        preds = model(X_batch)
        train_loss = criterion(preds, Y_batch)

        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()

    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            preds = model(X_batch)
            val_loss = criterion(preds, Y_batch)
            running_val_loss += val_loss.item()

    avg_train_loss = running_train_loss / len(train_dataloader)
    avg_val_loss = running_val_loss / len(val_dataloader)
    
    print(f'Epoch {epoch}/{n_epochs-1}, Train Loss: {avg_train_loss:.6e}, Val Loss: {avg_val_loss:.6e}')

    train_losses[epoch] = avg_train_loss
    val_losses[epoch] = avg_val_loss


fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.semilogy(train_losses, label='Train loss')
ax.semilogy(val_losses, label='Val loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)
ax.legend()




