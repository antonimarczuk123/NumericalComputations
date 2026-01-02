# %% __________________________________________________________________
# Regresja liniowa.
# Autor: Antoni Marczuk

import numpy as np
import matplotlib.pyplot as plt


# %% __________________________________________________________________
# Przygotowanie danych uczących i walidujących.


# Funkcja do aproksymacji
Fun = lambda x: 13 * x[0,:] - 24 * x[1,:] + 35 * x[2,:] - 46

n_inputs = 3 # liczba wejść

# Generowanie próbek uczących i walidujących

N= 1000 # liczba próbek uczących

X_min = 0; X_max = 10
X = np.random.uniform(X_min, X_max, (n_inputs, N))

Y = Fun(X).reshape(1, N)
Y = Y + np.random.normal(0, 5, Y.shape) # dodanie szumu


# %% __________________________________________________________________
# Uczenie.

wsp = 1e-8 # współczynnik regularyzacji
A = np.hstack((np.ones((N,1)), X.T))
W = np.linalg.solve(A.T @ A + wsp * np.eye(n_inputs + 1), A.T @ Y.T)

b = W[0]
W = W[1:].T


# %% __________________________________________________________________
# Wykresy diagnostyczne dopasowania modelu.

# --- Ocena modelu ---

Ymod = b + W @ X

MSE = np.mean((Y - Ymod) ** 2)
var_norm_MSE = MSE / np.var(Y)

print(f"\nMSE = {MSE:e},  var-norm-MSE = {var_norm_MSE:e}")

#  --- Wykres dopasowania (prawdziwy Y) vs (przewidywany Y) ---

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(Y, Ymod, s=4)
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
ax.set_title("Parity Plot for Linear Regression")
ax.set_xlabel("True Values")
ax.set_ylabel("Model Values")
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()



