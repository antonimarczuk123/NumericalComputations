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

Fun = lambda x: 100 * (torch.sin(x[:,0] * x[:,1]) + torch.cos(x[:,1] + x[:,0]))

n_inputs = 2
n_outputs = 1

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

X_min = 0
X_max = 10

X_train = torch.rand(n_train, n_inputs) * (X_max - X_min) + X_min
Y_train = Fun(X_train).reshape(n_train, n_outputs)

Y_min = Y_train.min() # minimalna wartość Y w zbiorze uczącym
Y_max = Y_train.max() # maksymalna wartość Y w zbiorze uczącym

X_train = (X_train - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_train = (Y_train - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

X_val = torch.rand(n_val, n_inputs) * (X_max - X_min) + X_min
Y_val = Fun(X_val).reshape(n_val, n_outputs)

X_val = (X_val - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_val = (Y_val - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

# -----

train_dataset = TensorDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
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

model.eval()
with torch.no_grad():
    train_exact = Y_train.cpu().numpy()
    train_preds = model(X_train).cpu().numpy()

    val_exact = Y_val.cpu().numpy()
    val_preds = model(X_val).cpu().numpy()

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.scatter(train_exact, train_preds, s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Train set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.scatter(val_exact, val_preds, s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Val set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()




