# %% __________________________________________________________________
# MLP z dowolną liczbą warstw ukrytych.
# Funkcje aktywacji: relu dla warstw ukrytych, liniowa dla wyjściowej.
# Uczenie metodą losowego przeszukiwania.
# Autor: Antoni Marczuk

# Na razie działa to bardzo kiepsko - wymaga dalszych prac!

import numpy as np
import matplotlib.pyplot as plt


# %% __________________________________________________________________
# Przygotowanie danych uczących i walidujących.


# Funkcja do aproksymacji
Fun = lambda x: 1000 * np.sin(x[0,:]) * np.cos(x[1,:]) / np.exp(0.01 * (x[0,:] + x[1,:]))

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_hidden = [10 for _ in range(10)] # liczba neuronów w warstwach ukrytych
n_outputs = 1 # liczba wyjść (nie zmianiać, bo kod zakłada 1 wyjście)

m = len(n_hidden) # liczba warstw ukrytych

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
# Inicjalizacja wag i poprzednich kroków minimalizacji.


# zerowa inicjalizacja biasów

b = [None] * (m + 1)
b[0] = np.zeros((n_hidden[0], 1))

for i in range(1, m):
    b[i] = np.zeros((n_hidden[i], 1))

b[m] = np.zeros((n_outputs, 1))

# zerowa inicjalizacja wag

W = [None] * (m + 1)
W[0] = np.zeros((n_hidden[0], n_inputs))

for i in range(1, m):
    W[i] = np.zeros((n_hidden[i], n_hidden[i-1]))

W[m] = np.zeros((n_outputs, n_hidden[-1]))


# %% __________________________________________________________________
# Uczenie sieci metodą losowego przeszukiwania.

max_epochs = 5000 # maksymalna liczba epok
sigma = 0.1      # odchylenie standardowe szumu dodawanego do biasów w pierwszej warstwie

# deklaracja potrzebnych tablic

b_new = [np.zeros_like(bi) for bi in b]
W_new = [np.zeros_like(Wi) for Wi in W]

MSEtrainTab = np.zeros(max_epochs + 1)
MSEvalTab = np.zeros(max_epochs + 1)

# Wyjściowe wartości MSE na zbiorze uczącym i walidującym

Yhat_train = X_train
for i in range(m):
    Yhat_train = np.maximum(0, W[i] @ Yhat_train + b[i])
Yhat_train = W[-1] @ Yhat_train + b[-1]

Yhat_val = X_val
for i in range(m):
    Yhat_val = np.maximum(0, W[i] @ Yhat_val + b[i])
Yhat_val = W[-1] @ Yhat_val + b[-1]

MSEtrain = np.mean((Yhat_train - Y_train) ** 2)
MSEval = np.mean((Yhat_val - Y_val) ** 2)

MSEtrainTab[0] = MSEtrain
MSEvalTab[0] = MSEval

# uczenie sieci
for epoch in range(max_epochs):

    b_new[0] = b[0] + np.random.normal(0, sigma, (n_hidden[0], 1))

    for i in range(1, m):
        b_new[i] = b[i] + np.random.normal(0, sigma, (n_hidden[i], 1))

    b_new[m] = b[m] + np.random.normal(0, sigma, (n_outputs, 1))

    W_new[0] = W[0] + np.random.normal(0, sigma, (n_hidden[0], n_inputs))
    for i in range(1, m):
        W_new[i] = W[i] + np.random.normal(0, sigma, (n_hidden[i], n_hidden[i-1]))

    W_new[m] = W[m] + np.random.normal(0, sigma, (n_outputs, n_hidden[-1]))

    # Obliczenie MSE na zbiorze uczącym i walidującym po epoce

    Yhat_train = X_train
    for i in range(m):
        Yhat_train = np.maximum(0, W_new[i] @ Yhat_train + b_new[i])
    Yhat_train = W_new[-1] @ Yhat_train + b_new[-1]

    Yhat_val = X_val
    for i in range(m):
        Yhat_val = np.maximum(0, W_new[i] @ Yhat_val + b_new[i])
    Yhat_val = W_new[-1] @ Yhat_val + b_new[-1]

    MSEtrain_new = np.mean((Yhat_train - Y_train) ** 2)
    MSEval_new = np.mean((Yhat_val - Y_val) ** 2)

    if MSEval_new < MSEval:
        # zaakceptuj nowe wagi i biasy
        for i in range(m + 1):
            b[i] = b_new[i]
            W[i] = W_new[i]

        MSEtrain = MSEtrain_new
        MSEval = MSEval_new

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
    Yhat_train = np.maximum(0, W[i] @ Yhat_train + b[i])
Yhat_train = W[-1] @ Yhat_train + b[-1]

Yhat_val = X_val
for i in range(m):
    Yhat_val = np.maximum(0, W[i] @ Yhat_val + b[i])
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
        

