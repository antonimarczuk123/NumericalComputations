# %% __________________________________________________________________
# MLP z dowolną liczbą warstw ukrytych.
# Uczenie metodą SGD + Nesterov momentum.
# Autor: Antoni Marczuk


import numpy as np
import matplotlib.pyplot as plt


# %% __________________________________________________________________
# Przygotowanie danych uczących i walidujących.


# Funkcja do aproksymacji
Fun = lambda x: np.sin(x[0,:]) + np.cos(x[1,:])

n_inputs = 2 # liczba wejść
n_hidden = [10 for _ in range(20)] # liczba neuronów w warstwach ukrytych
n_outputs = 1 # liczba wyjść

m = len(n_hidden) # liczba warstw ukrytych

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

# Generowanie próbek uczących i walidujących
X_min = 0; X_max = 10

X_train = np.random.uniform(X_min, X_max, (n_inputs, n_train))
Y_train = Fun(X_train).reshape(n_outputs, n_train)
X_train = (X_train - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]

X_val = np.random.uniform(X_min, X_max, (n_inputs, n_val))
Y_val = Fun(X_val).reshape(n_outputs, n_val)
X_val = (X_val - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]


# %% __________________________________________________________________
# Inicjalizacja wag i poprzednich kroków minimalizacji.


# Losowa inicjalizacja biasów
b = [None] * (m + 1)
b[0] = np.random.uniform(-1, 1, (n_hidden[0], 1))

for i in range(1, m):
    b[i] = np.random.uniform(-1, 1, (n_hidden[i], 1))

b[m] = np.random.uniform(-1, 1, (n_outputs, 1))

# Losowa inicjalizacja wag
W = [None] * (m + 1)
W[0] = np.random.uniform(-1, 1, (n_hidden[0], n_inputs))

for i in range(1, m):
    W[i] = np.random.uniform(-1, 1, (n_hidden[i], n_hidden[i-1]))

W[m] = np.random.uniform(-1, 1, (n_outputs, n_hidden[-1]))

# zerowa inicjalizacja poprzednich kroków minimalizacji
p_b_old = [np.zeros_like(bi) for bi in b]
p_W_old = [np.zeros_like(Wi) for Wi in W]


# %% __________________________________________________________________
# Uczenie sieci metodą SGD + Nesterov momentum.

max_epochs = 500 # maksymalna liczba epok
learning_rate = 0.001 # współczynnik uczenia
momentum = 0.9 # współczynnik momentum
mb_size = 64 # rozmiar mini-batcha

# deklaracja potrzebnych tablic

idx = np.zeros((1, mb_size), dtype=int)
X = np.zeros((n_inputs, mb_size))
Y = np.zeros((n_outputs, mb_size))

b_look = [np.zeros_like(bi) for bi in b] # biasy do kroku look-ahead
W_look = [np.zeros_like(Wi) for Wi in W] # wagi do kroku look-ahead

Z = [np.zeros((ni, mb_size)) for ni in n_hidden]
V = [np.zeros((ni, mb_size)) for ni in n_hidden]

p_b = [np.zeros_like(bi) for bi in b] # bieżące kroki minimalizacji dla biasów
p_W = [np.zeros_like(Wi) for Wi in W] # bieżące kroki minimalizacji dla wag

# gradienty błędu względem sygnałów wejściowych do warstw
dL_dZ = [np.zeros_like(Zi) for Zi in Z] + [np.zeros_like(Y)]

dE_db = [np.zeros_like(bi) for bi in b] # gradienty błędu względem biasów
dE_dW = [np.zeros_like(Wi) for Wi in W] # gradienty błędu względem wag

MSEtrainTab = np.zeros((max_epochs+1, 1))
MSEvalTab = np.zeros((max_epochs+1, 1))

# Wyjściowe wartości MSE na zbiorze uczącym i walidującym

Yhat_train = X_train
for i in range(m):
    Yhat_train = np.tanh(W[i] @ Yhat_train + b[i])
Yhat_train = W[-1] @ Yhat_train + b[-1]

Yhat_val = X_val
for i in range(m):
    Yhat_val = np.tanh(W[i] @ Yhat_val + b[i])
Yhat_val = W[-1] @ Yhat_val + b[-1]

MSEtrain = np.mean((Yhat_train - Y_train) ** 2)
MSEval = np.mean((Yhat_val - Y_val) ** 2)

MSEtrainTab[0] = MSEtrain
MSEvalTab[0] = MSEval

# uczenie sieci
for epoch in range(max_epochs):
    for iter in range(100):
        idx = np.random.randint(0, n_train, size=mb_size)
        X = X_train[:, idx]
        Y = Y_train[:, idx]

        # Nesterov look-ahead
        b_look = [bi + momentum * p_b_old_i for bi, p_b_old_i in zip(b, p_b_old)]
        W_look = [Wi + momentum * p_W_old_i for Wi, p_W_old_i in zip(W, p_W_old)]

        # Forward pass
        Z[0] = W_look[0] @ X + b_look[0]
        V[0] = np.tanh(Z[0])

        for i in range(1, m):
            Z[i] = W_look[i] @ V[i-1] + b_look[i]
            V[i] = np.tanh(Z[i])

        Y_hat = W_look[m] @ V[m-1] + b_look[m]

        # Backward pass
        dL_dZ[m] = 2 * (Y_hat - Y)

        for i in range(m-1, -1, -1):
            dL_dZ[i] = (W_look[i+1].T @ dL_dZ[i+1]) * (1 - V[i] ** 2)

        # Gradienty
        dE_db[0] = np.mean(dL_dZ[0], axis=1, keepdims=True)
        dE_dW[0] = (dL_dZ[0] @ X.T) / mb_size

        for i in range(1, m+1):
            dE_db[i] = np.mean(dL_dZ[i], axis=1, keepdims=True)
            dE_dW[i] = (dL_dZ[i] @ V[i-1].T) / mb_size

        # Aktualizacja kroków minimalizacji
        p_b = [ momentum * p_b_old_i - learning_rate * dE_db_i 
            for p_b_old_i, dE_db_i in zip(p_b_old, dE_db) ]
        
        p_W = [ momentum * p_W_old_i - learning_rate * dE_dW_i 
            for p_W_old_i, dE_dW_i in zip(p_W_old, dE_dW) ]
        
        # Aktualizacja wag i biasów
        b = [ bi + p_b_i for bi, p_b_i in zip(b, p_b) ]
        W = [ Wi + p_W_i for Wi, p_W_i in zip(W, p_W) ]

        # Zapisanie poprzednich kroków minimalizacji
        p_b_old = p_b
        p_W_old = p_W

    # Obliczenie MSE na zbiorze uczącym i walidującym po epoce

    Yhat_train = X_train
    for i in range(m):
        Yhat_train = np.tanh(W[i] @ Yhat_train + b[i])
    Yhat_train = W[-1] @ Yhat_train + b[-1]

    Yhat_val = X_val
    for i in range(m):
        Yhat_val = np.tanh(W[i] @ Yhat_val + b[i])
    Yhat_val = W[-1] @ Yhat_val + b[-1]

    MSEtrain = np.mean((Yhat_train - Y_train) ** 2)
    MSEval = np.mean((Yhat_val - Y_val) ** 2)

    MSEtrainTab[epoch+1] = MSEtrain
    MSEvalTab[epoch+1] = MSEval

    print(f"\rEpoka [{epoch+1}/{max_epochs}]  MSE train: {MSEtrain:.6e}  MSE val: {MSEval:.6e}", end='')


# Wyświetlanie przebiegu uczenia

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

ax1.plot(MSEtrainTab)
ax1.set_yscale('log')
ax1.set_title("MSEtrain")
ax1.minorticks_on()  # włącza dodatkowe podziałki
ax1.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

ax2.plot(MSEvalTab)
ax2.set_yscale('log')
ax2.set_title("MSEval")
ax2.set_xlabel("Epochs")
ax2.minorticks_on()  # włącza dodatkowe podziałki
ax2.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

plt.tight_layout() # ładniej wyglądają wykresy

        
# %% __________________________________________________________________
# Wykresy diagnostyczne dopasowania modelu.

# --- Ocena modelu ---

Yhat_train = X_train
for i in range(m):
    Yhat_train = np.tanh(W[i] @ Yhat_train + b[i])
Yhat_train = W[-1] @ Yhat_train + b[-1]

Yhat_val = X_val
for i in range(m):
    Yhat_val = np.tanh(W[i] @ Yhat_val + b[i])
Yhat_val = W[-1] @ Yhat_val + b[-1]

MSEtrain = np.mean((Yhat_train - Y_train) ** 2)
MSEval = np.mean((Yhat_val - Y_val) ** 2)

var_norm_MSE_train = MSEtrain / np.var(Y_train)
var_norm_MSE_val = MSEval / np.var(Y_val)

print(f"\n(train) MSE = {MSEtrain:e},  var-norm-MSE = {var_norm_MSE_train:e}")
print(f"(val) MSE = {MSEval:e},  var-norm-MSE = {var_norm_MSE_val:e}")

#  --- Wykres dopasowania (prawdziwy Y) vs (przewidywany Y) ---

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(Y_train, Yhat_train, s=4)
ax1.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--')
ax1.set_title("Parity Plot for MLP on Training Set")
ax1.set_xlabel("True Values")
ax1.set_ylabel("Model Values")
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2.scatter(Y_val, Yhat_val, s=4)
ax2.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--')
ax2.set_title("Parity Plot for MLP on Validation Set")
ax2.set_xlabel("True Values")
ax2.set_ylabel("Model Values")
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()
        

