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

# Opis co robi skrypt:
# Ten skrypt implementuje wielowarstwową sieć neuronową (MLP) do aproksymacji funkcji
# za pomocą JAX. Dzięki wykorzystaniu JAX, skrypt korzysta z automatycznego różniczkowania
# oraz kompilacji JIT. Ponadto automatyczny gradient pozwala na aproksymację funkcji wielowymiarowych.
# Stosowana metoda optymalizacji to algorytm Levenberga-Marquardta.
# JAX operuje na funkcjach, stąd po przygotowaniu danych definiujemy funkcje, które później
# są wykorzystywane w procesie uczenia sieci. Dla małej sieci szybciej działa na CPU.


# %% =================================================================
# Przygotowanie danych

# Funkcja do aproksymacji
Fun = lambda x: 1000 * jnp.sin(x[0] * x[1]) + jnp.cos(x[1] + x[0])

vmap_Fun = vmap(Fun, in_axes=0, out_axes=0)

n_inputs = 2 # liczba wejść (misi być takie jak w Fun)
n_hidden = [10 for _ in range(10)] # liczba neuronów w warstwach ukrytych
n_outputs = 1 # liczba wyjść

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
        std_dev = jnp.sqrt(2.0 / net_size[i])
        key, subkey = jrd.split(key)
        w[i] = jrd.normal(subkey, (net_size[i + 1], net_size[i])) * std_dev

    params = {'w': w, 'b': b}

    p_b_old = [jnp.zeros_like(bi) for bi in b]
    p_w_old = [jnp.zeros_like(wi) for wi in w]

    vel_params_old = {'w': p_w_old, 'b': p_b_old}

    return params, vel_params_old, key

def mlp_forward(params, x):
    for i in range(m-2):
        x = jnp.dot(params['w'][i], x) + params['b'][i]
        x = jax.nn.tanh(x)
    x = jnp.dot(params['w'][m-2], x) + params['b'][m-2]
    return x

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

def train_step(params, vel_params_old, learning_rate, momentum, key, batch_size):
    key, subkey = jrd.split(key)
    idxs = jrd.randint(subkey, (batch_size,), minval=0, maxval=n_train)
    x_batch = X_train[idxs]
    y_batch = Y_train[idxs]

    params_lookup = tree_map(lambda p, v_old: p + momentum * v_old, params, vel_params_old)
    nesterov_grads = grad_batch_loss(params_lookup, x_batch, y_batch)

    vel_params_new = tree_map(
        lambda v_old, g: momentum * v_old - learning_rate * g,
        vel_params_old, nesterov_grads)
    
    params_new = tree_map(
        lambda p, v_new: p + v_new,
        params, vel_params_new)
    
    return params_new, vel_params_new, key

def N_train_steps(params, vel_params_old, learning_rate, momentum, key, batch_size, n_steps):
    def loop_body(i, carry):
        params, vel_params_old, key = carry
        params, vel_params_old, key = train_step(
            params, vel_params_old, learning_rate, momentum, key, batch_size)
        return (params, vel_params_old, key)

    params, vel_params_old, key = fori_loop(
        0, n_steps, loop_body, (params, vel_params_old, key))
    
    return (params, vel_params_old, key)

jit_N_train_steps = jit(N_train_steps, static_argnames=('batch_size', 'n_steps'))



# %% =================================================================
# Uczenie

max_epochs = 2000 # maksymalna liczba epok
max_iter = 3000 # maksymalna liczba iteracji na epokę
learning_rate = 0.001 # współczynnik uczenia
momentum = 0.9 # współczynnik momentum
mb_size = 64 # rozmiar mini-batcha

# params, vel_params_old, key = initialize_mlp(key)

train_losses = np.zeros(max_epochs)
val_losses = np.zeros(max_epochs)

start = time.time()

for epoch in range(max_epochs):
    params, vel_params_old, key = jit_N_train_steps(
        params, vel_params_old, learning_rate, momentum, key, mb_size, max_iter)
    
    train_loss = batch_loss(params, X_train, Y_train)
    val_loss = batch_loss(params, X_val, Y_val)
    
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss

    print(f"Epoch {epoch}/{max_epochs-1}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}")

print(f"Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}")

end = time.time()
print(f"Training time: {end - start} seconds")

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.semilogy(train_losses, label='Train loss')
ax.semilogy(val_losses, label='Val loss')
ax.set_title('Training and Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)
ax.legend()

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(Y_train, jit_vmap_mlp_forward(params, X_train), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Train set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.scatter(Y_val, jit_vmap_mlp_forward(params, X_val), s=4)
ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') # linia y=x
ax.set_title('Val set')
ax.set_xlabel('True values')
ax.set_ylabel('Predicted values')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.show()



