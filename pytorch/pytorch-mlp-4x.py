# %% ========================================================================
# Imports

import torch

import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'

# Podstawowym typem danych w PyTorch jest tensor, wyposażony w dodatkowe properties:
# .data - zawiera dane
# .grad - zawiera gradienty jeśli są obliczane albo None
# .grad_fn - zawiera iformacje o operacji, która stworzyła ten tensor
# .requires_grad - boolean, który mówi czy ten tensor powinien być śledzony pod kątem gradientów
# .is_leaf - boolean, który mówi czy ten tensor jest "liściem" w drzewie operacji


# %% ========================================================================
# Prepare data

Fun = lambda x: 100 * (torch.sin(x[:,0] * x[:,1]) + torch.cos(x[:,1] + x[:,0]))

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_outputs = 1 # liczba wyjść

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


# %% ========================================================================
# Define and initialize model

class myMLP():
    def __init__(self):
        super().__init__()

        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None

        self.n_train = X_train.shape[0]
        self.n_val = X_val.shape[0]

        n0 = 2 # input
        n1 = 50
        n2 = 50
        n3 = 50
        n4 = 50
        n5 = 1 # output

        self.W0 = torch.zeros(n0, n1, requires_grad=True)
        self.W1 = torch.zeros(n1, n2, requires_grad=True)
        self.W2 = torch.zeros(n2, n3, requires_grad=True)
        self.W3 = torch.zeros(n3, n4, requires_grad=True)
        self.W4 = torch.zeros(n4, n5, requires_grad=True)

        self.b0 = torch.zeros(n1, requires_grad=True)
        self.b1 = torch.zeros(n2, requires_grad=True)
        self.b2 = torch.zeros(n3, requires_grad=True)
        self.b3 = torch.zeros(n4, requires_grad=True)
        self.b4 = torch.zeros(n5, requires_grad=True)

        self.params = [
            self.W0, self.W1, self.W2, self.W3, self.W4,
            self.b0, self.b1, self.b2, self.b3, self.b4
        ]

        self.optimizer = torch.optim.Adam(self.params, lr=0.001)

    def load_data(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

    def initialize_weights(self):
        with torch.no_grad():
            self.W0.normal_(0, torch.sqrt(torch.tensor(2.0 / self.W0.shape[0])))
            self.W1.normal_(0, torch.sqrt(torch.tensor(2.0 / self.W1.shape[0])))
            self.W2.normal_(0, torch.sqrt(torch.tensor(2.0 / self.W2.shape[0])))
            self.W3.normal_(0, torch.sqrt(torch.tensor(2.0 / self.W3.shape[0])))
            self.W4.normal_(0, torch.sqrt(torch.tensor(2.0 / self.W4.shape[0])))

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def forward_pass(self, x):
        h1 = torch.nn.functional.gelu(x @ self.W0 + self.b0)
        h2 = torch.nn.functional.gelu(h1 @ self.W1 + self.b1)
        h3 = torch.nn.functional.gelu(h2 @ self.W2 + self.b2)
        h4 = torch.nn.functional.gelu(h3 @ self.W3 + self.b3)
        y = h4 @ self.W4 + self.b4
        return y
    
    def loss(self, x, y):
        y_pred = self.forward_pass(x)
        return torch.mean((y_pred - y) ** 2)
    
    def train(self, n_steps, batch_size):
        for step in range(n_steps):
            idx = torch.randint(0, self.n_train, (batch_size,))
            x_batch = self.X_train[idx]
            y_batch = self.Y_train[idx]

            self.zero_grad()
            loss = self.loss(x_batch, y_batch)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            train_loss = self.loss(self.X_train, self.Y_train).item()
            val_loss = self.loss(self.X_val, self.Y_val).item()

        return train_loss, val_loss
    
mlp = myMLP()
mlp.load_data(X_train, Y_train, X_val, Y_val)
mlp.initialize_weights()
        

# %% ========================================================================
# Train model

n_epochs = 100
n_steps = 1000
batch_size = 64

train_losses = np.zeros(n_epochs)
val_losses = np.zeros(n_epochs)

for epoch in range(n_epochs):
    train_loss, val_loss = mlp.train(n_steps, batch_size)
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss
    print(f'Epoch {epoch}/{n_epochs-1}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}')

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






