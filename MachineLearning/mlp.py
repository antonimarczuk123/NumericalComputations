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
import matplotlib.pyplot as plt
import numpy as np

# Ustawienie urządzenia do obliczeń (CPU lub GPU)
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
jax.config.update("jax_default_device", cpu)



# %% =================================================================
# Przygotowanie danych

# Funkcja do aproksymacji
Fun = lambda x: 1000 * jnp.sin(x[0]) + jnp.cos(x[1])
vmap_Fun = vmap(Fun, in_axes=0, out_axes=0)

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_hidden = [10, 20, 10] # liczba neuronów w warstwach ukrytych
n_outputs = 1 # liczba wyjść (nie zmianiać, bo kod zakłada 1 wyjście)

net_size = [n_inputs] + n_hidden + [n_outputs]  # rozmiary warstw sieci
m = len(net_size)  # liczba warstw sieci

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

X_train = (X_train - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]
Y_train = (Y_train - Y_min) / (Y_max - Y_min)  # Przeskalowanie do [0, 1]

key, subkey = jrd.split(key)
X_val = jrd.uniform(subkey, (n_val, n_inputs), minval=X_min, maxval=X_max)
Y_val = vmap_Fun(X_val).reshape(n_val, n_outputs)

X_val = (X_val - X_min) / (X_max - X_min)  # Przeskalowanie do [0, 1]
Y_val = (Y_val - Y_min) / (Y_max - Y_min)  # Przeskalowanie do [0, 1]



# %% =================================================================
# Funkcje 

def initialize_mlp(key):
    b = [None] * (m - 1)
    for i in range(m - 1):
        b[i] = jnp.zeros((net_size[i + 1],))

    w = [None] * (m - 1)
    for i in range(m - 1):
        limit = jnp.sqrt(6 / (net_size[i] + net_size[i + 1]))
        key, subkey = jrd.split(key)
        w[i] = jrd.uniform(
            subkey, (net_size[i + 1], net_size[i]), minval=-limit, maxval=limit)

    params = {'w': w, 'b': b}

    p_b_old = [jnp.zeros_like(bi) for bi in b]
    p_w_old = [jnp.zeros_like(wi) for wi in w]

    vel_params_old = {'w': p_w_old, 'b': p_b_old}

    return params, vel_params_old, key

def mlp_forward(params, x):
    v = x
    for i in range(m-2):
        v = jnp.dot(params['w'][i], v) + params['b'][i]
        v = jax.nn.tanh(v)
    v = jnp.dot(params['w'][m-2], v) + params['b'][m-2]
    return v

jit_mlp_forward = jit(mlp_forward)
vmap_mlp_forward = vmap(mlp_forward, in_axes=(None, 0), out_axes=0)
jit_vmap_mlp_forward = jit(vmap_mlp_forward)

def single_loss(params, x, y):
    y_pred = mlp_forward(params, x)
    return jnp.sum((y_pred - y) ** 2)

vmap_single_loss = vmap(single_loss, in_axes=(None, 0, 0), out_axes=0)

def batch_loss(params, x_batch, y_batch):
    losses = vmap_single_loss(params, x_batch, y_batch)
    return jnp.mean(losses)

jit_batch_loss = jit(batch_loss)
grad_batch_loss = grad(batch_loss)

def train_step(params, vel_params_old, x_batch, y_batch, learning_rate, momentum):

    params_lookup = tree_map(lambda p, v_old: p + momentum * v_old, params, vel_params_old)
    nesterov_grads = grad_batch_loss(params_lookup, x_batch, y_batch)

    vel_params_new = tree_map(
        lambda v_old, g: momentum * v_old - learning_rate * g,
        vel_params_old, nesterov_grads)
    
    params_new = tree_map(
        lambda p, v_new: p + v_new,
        params, vel_params_new)
    
    return params_new, vel_params_new

def get_batches(X, Y, batch_size, key):
    n_samples = X.shape[0]
    key, subkey = jrd.split(key)
    permutation = jrd.permutation(subkey, n_samples)
    X, Y = X[permutation], Y[permutation]
    
    num_of_batches = n_samples // batch_size
    num_samples_to_use = num_of_batches * batch_size
    
    X_batches = X[:num_samples_to_use].reshape(num_of_batches, batch_size, -1)
    Y_batches = Y[:num_samples_to_use].reshape(num_of_batches, batch_size, -1)
    
    return X_batches, Y_batches, num_of_batches, key

def one_epoch(params, vel_params_old, X_train, 
            Y_train, batch_size, learning_rate, momentum, key):
    
    X_batches, Y_batches, num_of_batches, key = get_batches(
        X_train, Y_train, batch_size, key)
    
    def body_fun(i, carry):
        params, vel_params_old = carry
        x_batch = X_batches[i]
        y_batch = Y_batches[i]
        params_new, vel_params = train_step(
            params, vel_params_old, x_batch, y_batch, learning_rate, momentum)
        
        return (params_new, vel_params)
    
    params_new, vel_params = fori_loop(
        0, num_of_batches, body_fun, (params, vel_params_old))
    
    return params_new, vel_params, key

jit_one_epoch = jit(one_epoch, static_argnames=('batch_size',))



# %% =================================================================
# Uczenie

max_epochs = 5000 # maksymalna liczba epok
learning_rate = 0.001 # współczynnik uczenia
momentum = 0.9 # współczynnik momentum
mb_size = 64 # rozmiar mini-batcha

params, vel_params_old, key = initialize_mlp(key)

train_losses = np.zeros(max_epochs)
val_losses = np.zeros(max_epochs)

for epoch in range(max_epochs):
    params, vel_params_old, key = jit_one_epoch(
        params, vel_params_old, X_train, Y_train, mb_size, learning_rate, momentum, key)
    
    train_loss = jit_batch_loss(params, X_train, Y_train)
    val_loss = jit_batch_loss(params, X_val, Y_val)
    
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss
    
    if epoch % 50 == 0 or epoch == max_epochs - 1:
        print(f"\r{epoch+1}/{max_epochs}: Train Loss = {train_loss:.12f}, Val Loss = {val_loss:.12f}", end='')

fig1 = plt.figure()
fig1.tight_layout()
ax = fig1.add_subplot(111)

ax.semilogy(train_losses, label='Train loss')
ax.semilogy(val_losses, label='Val loss')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)
ax.legend()

fig2 = plt.figure()
fig2.tight_layout()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)

ax1.scatter(Y_train, jit_vmap_mlp_forward(params, X_train), s=0.5, alpha=0.5)
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2.scatter(Y_val, jit_vmap_mlp_forward(params, X_val), s=0.5, alpha=0.5)
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()

