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

import optax

import numpy as np
import matplotlib.pyplot as plt

# Ustawienie urządzenia do obliczeń (CPU lub GPU)
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
jax.config.update("jax_default_device", cpu)

# Włączenie/wyłączenie float64
jax.config.update("jax_enable_x64", True)


# %% =================================================================
# Przygotowanie danych do uczenia modelu rozmytego

# Parametry objektu
ro = 1e6; # g/m^3
roc = 1e6; # g/m^3
cp = 1; # cal/(g*K)
cpc = 1; # cal/(g*K)
k0 = 1e10; # 1/min
E_R = 8330.1; # 1/K
h = 130e6; # cal/kmol
a = 0.516e6; # cal/(K*m^3)
b = 0.5; # -

# Stałe procesu
V = 1; # m^3
Fin = 1; # m^3/min
F = 1; # m^3/min

# y = (CA, T)
# u = (CAin, FC)
# z = (Tin, TCin)

# x = (y, u, z) = (CA, T, CAin, FC, Tin, TCin)

Fun = lambda x: jnp.array([
    (1/V) * (Fin*x[2] - F*x[0] - V * k0 * jnp.exp(-E_R/x[1]) * x[0]),
    (1/(V*ro*cp)) * (Fin*ro*cp*x[4] - F*ro*cp*x[1] + V*h*k0*jnp.exp(-E_R/x[1])*x[0] - (a * x[3]**(b+1) / (x[3] + (a * x[3]**b) / (2 * roc * cpc))) * (x[1] - x[5]))
])

vmap_Fun = vmap(Fun, in_axes=0, out_axes=0)

n_inputs = 6 # liczba argumentów funkcji
n_outputs = 2 # liczba wartości funkcji

n_train = 10000 # liczba próbek uczących
n_val = 3000   # liczba próbek walidujących

# Inicjalizacja generatora liczb losowych
seed = round(time.time_ns() % 1e6)
key = jrd.key(seed)

CA_min = 0.06; CA_max = 0.26
T_min = 300; T_max = 500
CAin_min = 1.5; CAin_max = 2.5
FC_min = 10; FC_max = 20
Tin_min = 300; Tin_max = 400
TCin_min = 250; TCin_max = 350

min_X = jnp.array([CA_min, T_min, CAin_min, FC_min, Tin_min, TCin_min])
max_X = jnp.array([CA_max, T_max, CAin_max, FC_max, Tin_max, TCin_max])

# Próbki uczące
key, subkey = jrd.split(key)
X_train_raw = jrd.uniform(subkey, (n_train, n_inputs), minval=min_X, maxval=max_X)
Y_train_raw = vmap_Fun(X_train_raw)

# Próbki walidujące
key, subkey = jrd.split(key)
X_val_raw = jrd.uniform(subkey, (n_val, n_inputs), minval=min_X, maxval=max_X)
Y_val_raw = vmap_Fun(X_val_raw)

# --- NORMALIZACJA MIN-MAX ---
min_Y = jnp.min(Y_train_raw, axis=0)
max_Y = jnp.max(Y_train_raw, axis=0)

X_train = (X_train_raw - min_X) / (max_X - min_X)
Y_train = (Y_train_raw - min_Y) / (max_Y - min_Y)

X_val = (X_val_raw - min_X) / (max_X - min_X)
Y_val = (Y_val_raw - min_Y) / (max_Y - min_Y)


# %% =================================================================
# Inicjalizacja modelu TS i funkcji do uczenia modelu rozmytego

# Zmienną x0 rozmywam na zbiory: A1, A2, A3, ...
# Zmienną x1 rozmywam na zbiory: B1, B2, B3, ...
# Zbiory są gaussowskie z parametrami (mean, std).

# Reguły TS z funkcjami liniowymi w części THEN:
# R0: IF x0 is A1 AND x1 is B1 THEN y = A1 * [x0; x1] + B1 * [x2; x3] + E1 * [x4; x5]
# R1: IF x0 is A1 AND x1 is B2 THEN y = A2 * [x0; x1] + B2 * [x2; x3] + E2 * [x4; x5]
# R2: IF x0 is A1 AND x1 is B3 THEN y = A3 * [x0; x1] + B3 * [x2; x3] + E3 * [x4; x5]
# itd.

def initialize_ts(key):
    params = [] # wagi w poszczególnych funkcjach oraz parametry zbiorów rozmytych
    
    # Inicjalizacja zbiorów rozmytych dla y0
    n0 = 5 # liczba zbiorów rozmytych dla y0
    y0_means = jnp.linspace(0.0, 1.0, n0)
    dist = y0_means[1] - y0_means[0]
    y0_stds = jnp.full((n0,), 0.6 * dist)
    
    # Inicjalizacja zbiorów rozmytych dla y1
    n1 = 5 # liczba zbiorów rozmytych dla y1
    y1_means = jnp.linspace(0.0, 1.0, n1)
    dist = y1_means[1] - y1_means[0]
    y1_stds = jnp.full((n1,), 0.6 * dist)

    # Inicjalizacja wag dla reguł TS
    n_rules = n0 * n1 # liczba reguł TS
    key, subkey = jrd.split(key)
    A = jrd.normal(subkey, (n_rules, 2, 2)) # wagi dla y0 i y1
    key, subkey = jrd.split(key)
    B = jrd.normal(subkey, (n_rules, 2, 2)) # wagi dla u0 i u1
    key, subkey = jrd.split(key)
    E = jrd.normal(subkey, (n_rules, 2, 2)) # wagi dla z0 i z1

    params = {
        'y0_means': y0_means,
        'y0_stds': y0_stds,
        'y1_means': y1_means,
        'y1_stds': y1_stds,
        'A': A,
        'B': B,
        'E': E
    }
    vel_params_old = tree_map(lambda p: jnp.zeros_like(p), params)

    return params, vel_params_old, key

params, vel_params_old, key = initialize_ts(key) # Inicjalizacja modelu TS

optimizer = optax.adam(learning_rate=0.001) # Inicjalizacja optymalizatora
opt_state = optimizer.init(params) # Inicjalizacja stanu optymalizatora

def ts_forward(params, x):
    y0 = x[0]
    y1 = x[1]
    u0 = x[2]
    u1 = x[3]
    z0 = x[4]
    z1 = x[5]

    # Obliczanie stopni przynależności do zbiorów rozmytych
    mu_y0 = jnp.exp(-0.5 * ((y0 - params['y0_means']) / params['y0_stds']) ** 2)
    mu_y1 = jnp.exp(-0.5 * ((y1 - params['y1_means']) / params['y1_stds']) ** 2)

    # Obliczanie aktywacji reguł TS
    mu_rules = jnp.outer(mu_y0, mu_y1).flatten()  # Flatten to get a 1D array
    mu_rules = mu_rules / (jnp.sum(mu_rules) + 1e-8)  # Normalizacja aktywacji reguł

    # Obliczanie wyjścia TS
    y = jnp.array([[y0], [y1]])
    u = jnp.array([[u0], [u1]])
    z = jnp.array([[z0], [z1]])

    y_pred = jnp.sum(mu_rules[:, None, None] * (params['A'] @ y + params['B'] @ u + params['E'] @ z), axis=0)

    return y_pred.flatten()

vmap_ts_forward = vmap(ts_forward, in_axes=(None, 0), out_axes=0)

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

def train_step(params, opt_state, key, batch_size):
    key, subkey = jrd.split(key)
    idxs = jrd.randint(subkey, (batch_size,), minval=0, maxval=n_train)
    x_batch = X_train[idxs]
    y_batch = Y_train[idxs]

    grads = grad_batch_loss(params, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = tree_map(lambda p, u: p + u, params, updates)
    
    return params, opt_state, key

def N_train_steps(params, opt_state, key, batch_size, n_steps):
    def loop_body(i, carry):
        params, opt_state, key = carry
        params, opt_state, key = train_step(params, opt_state, key, batch_size)
        return (params, opt_state, key)

    params, opt_state, key = fori_loop(0, n_steps, loop_body, (params, opt_state, key))
    
    train_loss = batch_loss(params, X_train, Y_train)
    val_loss = batch_loss(params, X_val, Y_val)
    grad_norm = norm_grad_batch_loss(params, X_train, Y_train)

    return params, opt_state, key, train_loss, val_loss, grad_norm

jit_N_train_steps = jit(N_train_steps, static_argnames=('batch_size', 'n_steps'))

# we need it to keep the best parameters based on validation loss
best_val_loss = jnp.inf
best_params = None


# %% =================================================================
# Uczenie

max_epochs = 100 # maksymalna liczba epok
max_iter = 1000 # maksymalna liczba iteracji na epokę
mb_size = 64 # rozmiar mini-batcha

train_losses = np.zeros(max_epochs)
val_losses = np.zeros(max_epochs)
norms_of_grad = np.zeros(max_epochs)

start = time.time()

for epoch in range(max_epochs):
    params, opt_state, key, train_loss, val_loss, grad_norm = jit_N_train_steps(
        params, opt_state, key, mb_size, max_iter)
    
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss
    norms_of_grad[epoch] = grad_norm

    print(f"Epoch {epoch}/{max_epochs-1}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}, Grad Norm = {grad_norm:.6e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params

end = time.time()

print(f"Training time: {end - start} seconds")
print(f"Best val Loss = {batch_loss(best_params, X_val, Y_val):.6e}")

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

# fig2 = plt.figure()
# ax = fig2.add_subplot(111)
# ax.semilogy(norms_of_grad, label='Grad norm')
# ax.legend()
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Norm')
# ax.minorticks_on()
# ax.grid(True, which='major', linestyle='-')
# ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()


