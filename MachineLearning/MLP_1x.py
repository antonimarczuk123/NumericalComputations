# %% __________________________________________________________________
# MLP z 1 warstwą ukrytą.
# Na koniec douczenie sieci metodą ELM.
# Funckje aktywacji: tanh dla warstw ukrytych, liniowa dla wyjściowej.
# Autor: Antoni Marczuk

import numpy as np
import matplotlib.pyplot as plt


# %% __________________________________________________________________
# Przygotowanie danych uczących i walidujących.


# Funkcja do aproksymacji
Fun = lambda x: 1000 * np.sin(x[0,:]) * np.cos(x[1,:]) / np.exp(0.01 * (x[0,:] + x[1,:]))

n_inputs = 2 # liczba wejść
n_hidden = 500 # liczba neuronów ukrytych
n_outputs = 1 # liczba wyjść

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

# Generowanie próbek uczących i walidujących
X_min = 0
X_max = 10

X_train = np.random.uniform(X_min, X_max, (n_inputs, n_train))
Y_train = Fun(X_train).reshape(n_outputs, n_train)

Y_min = Y_train.min() # minimalna wartość Y w zbiorze uczącym
Y_max = Y_train.max() # maksymalna wartość Y w zbiorze uczącym

X_train = (X_train - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]
Y_train = (Y_train - Y_min) / (Y_max - Y_min)  # Przeskalowanie do [0, 1]

X_val = np.random.uniform(X_min, X_max, (n_inputs, n_val))
Y_val = Fun(X_val).reshape(n_outputs, n_val)

X_val = (X_val - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]
Y_val = (Y_val - Y_min) / (Y_max - Y_min)  # Przeskalowanie do [0, 1]


# %% __________________________________________________________________
# Inicjalizacja wag i biasów sieci.

# Losowa inicjalizacja wag i biasów

b1 = np.random.randn(n_hidden, 1)
W1 = np.random.randn(n_hidden, n_inputs)

b2 = np.random.randn(n_outputs, 1)
W2 = np.random.randn(n_outputs, n_hidden)

# zerowa inicjalizacja poprzednich kroków minimalizacji

p_b1_old = np.zeros(b1.shape)
p_W1_old = np.zeros(W1.shape)

p_b2_old = np.zeros(b2.shape)
p_W2_old = np.zeros(W2.shape)


# %% __________________________________________________________________
# Uczenie sieci metodą SGD + Nesterov momentum.

max_epochs = 200 # maksymalna liczba epok
learning_rate = 0.001 # współczynnik uczenia
momentum = 0.9 # współczynnik momentum
mb_size = 64 # rozmiar mini-batcha


# deklaracja potrzebnych tablic

idx = np.zeros((1, mb_size), dtype=int)
X = np.zeros((n_inputs, mb_size))
Y = np.zeros((n_outputs, mb_size))

# ---

# Z0 = V0 = X

Z1 = np.zeros((n_hidden, mb_size))
V1 = np.zeros((n_hidden, mb_size))

# Z2 = V2 = Y_hat
Y_hat = np.zeros((n_outputs, mb_size)) 

# ---

p_b1 = np.zeros(b1.shape)
p_W1 = np.zeros(W1.shape)

p_b2 = np.zeros(b2.shape)
p_W2 = np.zeros(W2.shape)

# ---

dL_Z2 = np.zeros(Y_hat.shape)
dL_Z1 = np.zeros(Z1.shape)

# ---

dE_db1 = np.zeros(b1.shape)
dE_dW1 = np.zeros(W1.shape)

dE_db2 = np.zeros(b2.shape)
dE_dW2 = np.zeros(W2.shape)

# ---

b1_look = np.zeros(b1.shape)
W1_look = np.zeros(W1.shape)

b2_look = np.zeros(b2.shape)
W2_look = np.zeros(W2.shape)

# ---

MSEtrainTab = np.zeros((max_epochs+1, 1))
MSEvalTab = np.zeros((max_epochs+1, 1))


# Wyjściowe wartości MSE na zbiorze uczącym i walidującym

Ymodel_train = b2 + W2 @ np.tanh( b1 + W1 @ X_train)
Ymodel_val =   b2 + W2 @ np.tanh( b1 + W1 @ X_val)

MSEtrain = np.mean((Ymodel_train - Y_train) ** 2)
MSEval = np.mean((Ymodel_val - Y_val) ** 2)

MSEtrainTab[0] = MSEtrain
MSEvalTab[0] = MSEval


# uczenie sieci metodą SGD+momentum
for i in range(max_epochs):
    for j in range(300):
        idx = np.random.randint(0, n_train, size=mb_size)
        X = X_train[:, idx]
        Y = Y_train[:, idx]

        # Nesterov lookahead
        b1_look = b1 + momentum * p_b1_old
        W1_look = W1 + momentum * p_W1_old
        b2_look = b2 + momentum * p_b2_old
        W2_look = W2 + momentum * p_W2_old

        # Forward pass
        Z1 = W1_look @ X + b1_look
        V1 = np.tanh( Z1) # f1 = tanh
        Y_hat = W2_look @ V1 + b2_look

        # Backward pass
        dL_Z2 = 2 * (Y_hat - Y)
        dL_Z1 = (W2_look.T @ dL_Z2) * (1 - V1 ** 2) # df1

        # Gradienty
        dE_db2 = np.mean(dL_Z2, axis=1, keepdims=True)
        dE_dW2 = (dL_Z2 @ V1.T) / mb_size
        dE_db1 = np.mean(dL_Z1, axis=1, keepdims=True)
        dE_dW1 = (dL_Z1 @ X.T) / mb_size

        # Aktualizacja kroków
        p_b2 = momentum * p_b2_old - learning_rate * dE_db2
        p_W2 = momentum * p_W2_old - learning_rate * dE_dW2
        p_b1 = momentum * p_b1_old - learning_rate * dE_db1
        p_W1 = momentum * p_W1_old - learning_rate * dE_dW1

        # Aktualizacja wag i biasów
        b2 += p_b2
        W2 += p_W2
        b1 += p_b1
        W1 += p_W1

        # Zapisanie poprzednich kroków
        p_b2_old = p_b2
        p_W2_old = p_W2
        p_b1_old = p_b1
        p_W1_old = p_W1

    Ymodel_train = b2 + W2 @ np.tanh( b1 + W1 @ X_train)
    Ymodel_val =   b2 + W2 @ np.tanh( b1 + W1 @ X_val)

    MSEtrain = np.mean((Ymodel_train - Y_train) ** 2)
    MSEval = np.mean((Ymodel_val - Y_val) ** 2)

    MSEtrainTab[i+1] = MSEtrain
    MSEvalTab[i+1] = MSEval

    print(f"\rEpoka [{i+1}/{max_epochs}]  MSE train: {MSEtrain:.6e}  MSE val: {MSEval:.6e}", end='')


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
# Wykresy diagnostyczne przed ELM.

# --- Ocena modelu ---

Ymodel_train = b2 + W2 @ np.tanh( b1 + W1 @ X_train)
Ymodel_val =   b2 + W2 @ np.tanh( b1 + W1 @ X_val)

MSEtrain = np.mean((Ymodel_train - Y_train) ** 2)
MSEval = np.mean((Ymodel_val - Y_val) ** 2)

var_norm_MSE_train = MSEtrain / np.var(Y_train)
var_norm_MSE_val = MSEval / np.var(Y_val)

print(f"\n(train) MSE = {MSEtrain:e},  var-norm-MSE = {var_norm_MSE_train:e}")
print(f"(val) MSE = {MSEval:e},  var-norm-MSE = {var_norm_MSE_val:e}")

#  --- Wykres dopasowania (prawdziwy Y) vs (przewidywany Y) ---

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(Y_train, Ymodel_train, s=4)
ax1.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--')
ax1.set_title("Parity Plot for MLP on Training Set")
ax1.set_xlabel("True Values")
ax1.set_ylabel("Model Values")
ax1.minorticks_on()  # włącza dodatkowe podziałki
ax1.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

ax2.scatter(Y_val, Ymodel_val, s=4)
ax2.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--')
ax2.set_title("Parity Plot for MLP on Validation Set")
ax2.set_xlabel("True Values")
ax2.set_ylabel("Model Values")
ax2.minorticks_on()  # włącza dodatkowe podziałki
ax2.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

plt.tight_layout() # ładniej wyglądają wykresy


# %% __________________________________________________________________
# Douczenie sieci metodą ELM.

lambda_reg = 1e-10  # współczynnik regularizacji

A = np.hstack(( np.ones((n_train, 1)), (np.tanh( b1 + W1 @ X_train)).T ))
W = np.linalg.solve(A.T @ A + lambda_reg * np.eye(n_hidden + 1), A.T @ Y_train.T)

b2v2 = W[0]
W2v2 = W[1:].reshape(1, n_hidden)


# %% __________________________________________________________________
# Ocena modelu po douczeniu metodą ELM.

# --- Ocena modelu przed douczeniem ---

Ymodel_train = b2v2 + W2v2 @ np.tanh( W1 @ X_train + b1)
Ymodel_val = b2v2 + W2v2 @ np.tanh( W1 @ X_val + b1)

MSEtrain = np.mean((Ymodel_train - Y_train) ** 2)
MSEval = np.mean((Ymodel_val - Y_val) ** 2)

var_norm_MSE_train = MSEtrain / np.var(Y_train)
var_norm_MSE_val = MSEval / np.var(Y_val)

print(f"\n(train) MSE = {MSEtrain:e},  var-norm-MSE = {var_norm_MSE_train:e}")
print(f"(val) MSE = {MSEval:e},  var-norm-MSE = {var_norm_MSE_val:e}")

#  --- Wykres dopasowania (prawdziwy Y) vs (przewidywany Y) ---

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(Y_train, Ymodel_train, s=4)
ax1.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--', lw=2)
ax1.set_title("Parity Plot for ELM on Training Set")
ax1.set_xlabel("True Values")
ax1.set_ylabel("Model Values")
ax1.minorticks_on()  # włącza dodatkowe podziałki
ax1.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

ax2.scatter(Y_val, Ymodel_val, s=4)
ax2.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--', lw=2)
ax2.set_title("Parity Plot for ELM on Validation Set")
ax2.set_xlabel("True Values")
ax2.set_ylabel("Model Values")
ax2.minorticks_on()  # włącza dodatkowe podziałki
ax2.grid(True, which='major', linestyle='-')   # grubsze linie dla głównych
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)   # cieńsze dla pomocniczych

