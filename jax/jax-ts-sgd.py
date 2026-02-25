# %% ==================================================================
# Importy

import time
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap
from jax import jit
from jax import grad
from jax.lax import fori_loop
from jax.tree_util import tree_map
from jax.tree_util import tree_reduce
import matplotlib.pyplot as plt
import numpy as np

# Ustawienie urządzenia do obliczeń (CPU lub GPU)
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
jax.config.update("jax_default_device", cpu)

# Włączenie/wyłączenie float64
jax.config.update("jax_enable_x64", False)


# %% =================================================================
# Przygotowanie danych

# Funkcja do aproksymacji
Fun = lambda x: 100 * (jnp.sin(x[0] + x[1]) + jnp.cos(x[1]))

vmap_Fun = vmap(Fun, in_axes=0, out_axes=0)

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_outputs = 1 # liczba wyjść

n_train = 10000 # liczba próbek uczących
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

# %% =================================================================
# Inicjalizacja modelu TS i funkcji 

# Zmienną x0 rozmywam na zbiory: A1, A2, A3, ...
# Zmienną x1 rozmywam na zbiory: B1, B2, B3, ...
# Zbiory są gaussowskie z parametrami (mean, std).

# Reguły TS:
# R0: IF x0 is A1 AND x1 is B1 THEN y = a0 * x0 + b0 * x1 + c0
# R1: IF x0 is A1 AND x1 is B2 THEN y = a1 * x0 + b1 * x1 + c1
# R2: IF x0 is A1 AND x1 is B3 THEN y = a2 * x0 + b2 * x1 + c2
# itd.

def initialize_ts(key):
    params = [] # wagi w poszczególnych funkcjach oraz parametry zbiorów rozmytych
    
    # Inicjalizacja zbiorów rozmytych dla x0
    n0 = 10 # liczba zbiorów rozmytych dla x0
    x0_means = jnp.linspace(-1, 1, n0)
    dist = x0_means[1] - x0_means[0]
    x0_stds = jnp.full((n0,), 0.6 * dist)

    # Inicjalizacja zbiorów rozmytych dla x1
    n1 = 10 # liczba zbiorów rozmytych dla x1
    x1_means = jnp.linspace(-1, 1, n1)
    dist = x1_means[1] - x1_means[0]
    x1_stds = jnp.full((n1,), 0.6 * dist)

    # Inicjalizacja wag dla reguł TS
    n_rules = n0 * n1 # liczba reguł TS
    key, subkey = jrd.split(key)
    a = jrd.normal(subkey, (n_rules,)) # wagi dla x0
    key, subkey = jrd.split(key)
    b = jrd.normal(subkey, (n_rules,)) # wagi dla x1
    key, subkey = jrd.split(key)
    c = jrd.normal(subkey, (n_rules,)) # wagi stałe

    params = {
        'x0_means': x0_means,
        'x0_stds': x0_stds,
        'x1_means': x1_means,
        'x1_stds': x1_stds,
        'a': a,
        'b': b,
        'c': c
    }
    vel_params_old = tree_map(lambda p: jnp.zeros_like(p), params)

    return params, vel_params_old, key

params, vel_params_old, key = initialize_ts(key) # Inicjalizacja modelu TS

def ts_forward(params, x):
    x0 = x[0]
    x1 = x[1]

    # Obliczanie stopni przynależności do zbiorów rozmytych
    mu_x0 = jnp.exp(-0.5 * ((x0 - params['x0_means']) / params['x0_stds']) ** 2)
    mu_x1 = jnp.exp(-0.5 * ((x1 - params['x1_means']) / params['x1_stds']) ** 2)

    # Obliczanie aktywacji reguł TS
    mu_rules = jnp.outer(mu_x0, mu_x1).flatten()  # Flatten to get a 1D array
    mu_rules = mu_rules / (jnp.sum(mu_rules) + 1e-8)  # Normalizacja aktywacji reguł

    # Obliczanie wyjścia TS
    y_pred = jnp.sum(mu_rules * (params['a'] * x0 + params['b'] * x1 + params['c']))

    return y_pred

vmap_ts_forward = vmap(ts_forward, in_axes=(None, 0), out_axes=0)
jit_vmap_ts_forward = jit(vmap_ts_forward)

def single_loss(params, x, y):
    y_pred = ts_forward(params, x)
    return jnp.sum((y_pred - y) ** 2)

vmap_single_loss = vmap(single_loss, in_axes=(None, 0, 0), out_axes=0)

def batch_loss(params, x_batch, y_batch):
    losses = vmap_single_loss(params, x_batch, y_batch)
    return jnp.mean(losses)

grad_batch_loss = grad(batch_loss)

def norm_grad_batch_loss(params, x_batch, y_batch):
    grad = grad_batch_loss(params, x_batch, y_batch)
    norm = tree_reduce(lambda acc, g: acc + jnp.sum(g ** 2), grad, initializer=0.0)
    norm = jnp.sqrt(norm)
    return norm

def train_step(params, vel_params_old, learning_rate, momentum, key, batch_size):
    key, subkey = jrd.split(key)
    idxs = jrd.randint(subkey, (batch_size,), minval=0, maxval=n_train)
    x_batch = X_train[idxs]
    y_batch = Y_train[idxs]

    params_nesterov_lookup = tree_map(lambda p, v_old: p + momentum * v_old, params, vel_params_old)
    nesterov_grads = grad_batch_loss(params_nesterov_lookup, x_batch, y_batch)

    vel_params_new = tree_map(
        lambda v_old, g: momentum * v_old - learning_rate * g,
        vel_params_old, nesterov_grads)
    
    params_new = tree_map(
        lambda p, v_new: p + v_new,
        params, vel_params_new)
    
    return params_new, vel_params_new, key

def N_train_steps(params, vel_params_old, learning_rate, momentum, key, 
                batch_size, n_steps):
    def loop_body(i, carry):
        params, vel_params_old, key = carry
        params, vel_params_old, key = train_step(
            params, vel_params_old, learning_rate, momentum, key, batch_size)
        return (params, vel_params_old, key)

    params, vel_params_old, key = fori_loop(
        0, n_steps, loop_body, (params, vel_params_old, key))
    
    train_loss = batch_loss(params, X_train, Y_train)
    val_loss = batch_loss(params, X_val, Y_val)
    grad_norm = norm_grad_batch_loss(params, X_train, Y_train)

    return params, vel_params_old, key, train_loss, val_loss, grad_norm

jit_N_train_steps = jit(N_train_steps, static_argnames=('batch_size', 'n_steps'))



# %% =================================================================
# Uczenie

max_epochs = 100 # maksymalna liczba epok
max_iter = 5000 # maksymalna liczba iteracji na epokę
learning_rate = 0.001 # współczynnik uczenia
momentum = 0.9 # współczynnik momentum
mb_size = 64 # rozmiar mini-batcha

train_losses = np.zeros(max_epochs)
val_losses = np.zeros(max_epochs)
norms_of_grad = np.zeros(max_epochs)

start = time.time()

for epoch in range(max_epochs):
    params, vel_params_old, key, train_loss, val_loss, grad_norm = jit_N_train_steps(
        params, vel_params_old, learning_rate, momentum, key, mb_size, max_iter)
    
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss
    norms_of_grad[epoch] = grad_norm

    print(f"Epoch {epoch}/{max_epochs-1}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}, Grad Norm = {grad_norm:.6e}")

print(f"Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}")

end = time.time()
print(f"Training time: {end - start} seconds")

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

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.semilogy(norms_of_grad, label='Grad norm')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Norm')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.scatter(Y_train, jit_vmap_ts_forward(params, X_train), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Train set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.scatter(Y_val, jit_vmap_ts_forward(params, X_val), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Val set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()



