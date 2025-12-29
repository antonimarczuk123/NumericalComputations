# %% __________________________________________________________________
# ELM - Extreme Learning Machine.
# Autor: Antoni Marczuk

import numpy as np
import matplotlib.pyplot as plt


# %% __________________________________________________________________
# Przygotowanie danych uczących i walidujących.

# Funkcja do aproksymacji
Fun = lambda x: np.sin(x[0,:]) * np.cos(x[1,:])

n_inputs = 2 # liczba wejść
n_hidden = 2000 # liczba neuronów ukrytych
n_outputs = 1 # liczba wyjść

fi = lambda x: np.tanh(x) # funkcja aktywacji neuronów ukrytych

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
# Losowa inicjalizacja wag pierwszej warstwy.

# losowa inicjalizacja wag i biasów
b1 = np.random.uniform(-10, 10, (n_hidden, 1))
W1 = np.random.uniform(-10, 10, (n_hidden, n_inputs))


# %% __________________________________________________________________
# Uczenie sieci.

lambda_reg = 0.0001  # współczynnik regularizacji

A = np.hstack(( np.ones((n_train, 1)), (fi(b1 + W1 @ X_train)).T ))
W = np.linalg.solve(A.T @ A + lambda_reg * np.eye(n_hidden + 1), A.T @ Y_train.T)

b2 = W[0]
W2 = W[1:].reshape(1, n_hidden)


# %% __________________________________________________________________
# Ocena modelu.

# --- Ocena modelu przed douczeniem ---

Ymodel_train = b2 + W2 @ fi(W1 @ X_train + b1)
Ymodel_val = b2 + W2 @ fi(W1 @ X_val + b1)

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



