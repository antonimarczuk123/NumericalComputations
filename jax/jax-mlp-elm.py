# %% ==================================================================
# Importy

import time
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt
import numpy as np

# Ustawienie urządzenia do obliczeń (CPU lub GPU)
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
jax.config.update("jax_default_device", cpu)

# Włączenie/wyłączenie float64
jax.config.update("jax_enable_x64", True)


# %% =================================================================
# Przygotowanie danych

# Funkcja do aproksymacji
Fun = lambda x: 1000 * jnp.sin(x[0] + x[1]) + jnp.cos(x[1])

vmap_Fun = vmap(Fun, in_axes=0, out_axes=0)

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_hidden = 500 # liczba neuronów w warstwie ukrytej
n_outputs = 1 # liczba wyjść (zakładam jedno wyjście)

n_train = 5000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

# Inicjalizacja generatora liczb losowych
seed = round(time.time_ns() % 1e6)
key = jrd.key(seed)

# Generowanie próbek uczących i walidujących
X_min = 0
X_max = 10

key, subkey = jrd.split(key)
X_train = jrd.uniform(subkey, (n_train, n_inputs), minval=X_min, maxval=X_max)
Y_train = vmap_Fun(X_train).reshape(n_train, n_outputs)

Y_min = Y_train.min() # minimalna wartość Y w zbiorze uczącym
Y_max = Y_train.max() # maksymalna wartość Y w zbiorze uczącym

X_train = (X_train - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_train = (Y_train - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

key, subkey = jrd.split(key)
X_val = jrd.uniform(subkey, (n_val, n_inputs), minval=X_min, maxval=X_max)
Y_val = vmap_Fun(X_val).reshape(n_val, n_outputs)

X_val = (X_val - X_min) / (X_max - X_min) * 2 - 1  # Przeskalowanie do [-1, 1]
Y_val = (Y_val - Y_min) / (Y_max - Y_min) * 2 - 1  # Przeskalowanie do [-1, 1]

varY_train = jnp.var(Y_train)
varY_val = jnp.var(Y_val)



# %% =================================================================
# Inicjalizacja funkcji 

def initialize_first_layer(key, w0_std_dev, b0_std_dev):
    key, subkey1, subkey2 = jrd.split(key, 3)
    w0 = jrd.normal(subkey1, (n_hidden, n_inputs)) * w0_std_dev
    b0 = jrd.normal(subkey2, (n_hidden,)) * b0_std_dev
    return w0, b0, key

def forward_first_layer(w0, b0, x):
    return jnp.tanh(w0 @ x + b0)

vmap_forward_first_layer = vmap(forward_first_layer, in_axes=(None, None, 0), out_axes=0)

def forward_second_layer(w1, b1, x):
    return w1 @ x + b1

def mlp_forward(w0, b0, w1, b1, x):
    v1 = forward_first_layer(w0, b0, x)
    return forward_second_layer(w1, b1, v1)

vmap_mlp_forward = vmap(mlp_forward, in_axes=(None, None, None, None, 0), out_axes=0)

def single_loss(w0, b0, w1, b1, x, y):
    y_pred = mlp_forward(w0, b0, w1, b1, x)
    return (y_pred - y) ** 2

vmap_single_loss = vmap(single_loss, in_axes=(None, None, None, None, 0, 0), out_axes=0)

def batch_loss(w0, b0, w1, b1, x_batch, y_batch):
    losses = vmap_single_loss(w0, b0, w1, b1, x_batch, y_batch)
    return jnp.mean(losses)

def train(w0_std_dev, b0_std_dev, lambda_reg, key):
    w0, b0, key = initialize_first_layer(key, w0_std_dev, b0_std_dev)

    A = jnp.hstack(( jnp.ones((n_train, 1)), vmap_forward_first_layer(w0, b0, X_train) ))
    W = jnp.linalg.solve(A.T @ A + lambda_reg * jnp.eye(n_hidden + 1), A.T @ Y_train)

    b1 = W[0]
    w1 = W[1:].reshape(1, n_hidden)

    MSE_train = batch_loss(w0, b0, w1, b1, X_train, Y_train)
    MSE_val = batch_loss(w0, b0, w1, b1, X_val, Y_val)

    MSE_norm_train = MSE_train / varY_train
    MSE_norm_val = MSE_val / varY_val

    return w0, b0, w1, b1, MSE_norm_train, MSE_norm_val, key

jit_train = jit(train)



# %% =================================================================
# Uczenie

max_iter = 30
w0_std_dev = 2.5
b0_std_dev = 3.0
lambda_reg = 1e-8

w0 = None
b0 = None
w1 = None
b1 = None
MSE_norm_train = jnp.inf
MSE_norm_val = jnp.inf

start = time.time()

for iter in range(max_iter):
    w0_new, b0_new, w1_new, b1_new, MSE_norm_train_new, MSE_norm_val_new, key = jit_train(
        w0_std_dev, b0_std_dev, lambda_reg, key)
    
    if MSE_norm_val_new < MSE_norm_val:
        w0 = w0_new
        b0 = b0_new
        w1 = w1_new
        b1 = b1_new
        MSE_norm_train = MSE_norm_train_new
        MSE_norm_val = MSE_norm_val_new
    
    print(f"{iter}/{max_iter-1}, MSE Norm Train = {MSE_norm_train_new:.6e}, MSE Norm Val = {MSE_norm_val_new:.6e}")
    
end = time.time()
print("--------------------------------------------------")
print(f"Training time: {end - start} seconds")
print(f"MSE Norm Train = {MSE_norm_train:.6e}, MSE Norm Val = {MSE_norm_val:.6e}")

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(Y_train, vmap_mlp_forward(w0, b0, w1, b1, X_train), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Train set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.scatter(Y_val, vmap_mlp_forward(w0, b0, w1, b1, X_val), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Val set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()



