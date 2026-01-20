# _____________________________________________________________
# Importy

import jax
from jax import numpy as jnp

import matplotlib.pyplot as plt
# plt.style.use('dark_background')

# _____________________________________________________________
# Przygotowanie danych uczących i walidujących.

n_inputs = 2
n_outputs = 1

def Fun(x):
    """ x in R^2, y in R """
    y = x[0] + x[1]
    return y

Fun_vec = jax.vmap(Fun)

# generowanie danych uczących i walidujących
n_train = 10000
n_val = 10000

Xmin = 0.0
Xmax = 10.0

X_train = jax.random.uniform(jax.random.PRNGKey(0), (n_train, n_inputs), minval=Xmin, maxval=Xmax)
Y_train = Fun_vec(X_train)

Ymin = jnp.min(Y_train)
Ymax = jnp.max(Y_train)

X_val = jax.random.uniform(jax.random.PRNGKey(1), (n_val, n_inputs), minval=Xmin, maxval=Xmax)
Y_val = Fun_vec(X_val)

# normalizacja danych
X_train = (X_train - Xmin) / (Xmax - Xmin)
Y_train = (Y_train - Ymin) / (Ymax - Ymin)
X_val = (X_val - Xmin) / (Xmax - Xmin)
Y_val = (Y_val - Ymin) / (Ymax - Ymin)

# wizualizacja danych uczących i walidujących
# fig = plt.figure()

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter3D(X_train[:,0], X_train[:,1], Y_train, color='green')
# ax1.set_xlabel('X1')
# ax1.set_ylabel('X2')
# ax1.set_zlabel('Y')
# ax1.set_title('Dane uczące')

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter3D(X_val[:,0], X_val[:,1], Y_val, color='red')
# ax2.set_xlabel('X1')
# ax2.set_ylabel('X2')
# ax2.set_zlabel('Y')
# ax2.set_title('Dane walidujące')

# plt.show()

# _____________________________________________________________
# Definicja funkcji dla MLP

""" def mlp(params, x_single) """
def mlp(params, x_single):
    y = x_single
    for W, b in params[:-1]:
        y = jax.nn.tanh(b + W @ y)
        
    W, b = params[-1]
    y = b + W @ y
    return y


""" def mlp_batch(params, x_batch) """
mlp_batch = jax.vmap(mlp, in_axes=(None, 0))


""" def loss_batch(params, x_batch, y_batch) """
def loss_batch(params, x_batch, y_batch):
    y_pred_batch = mlp_batch(params, x_batch)
    return jnp.mean((y_batch - y_pred_batch) ** 2)

def loss_val_full(params):
    return loss_batch(params, X_val, Y_val)

loss_val_full_jit = jax.jit(loss_val_full)

""" def grad_params_loss_batch(params, x_batch, y_batch) """
grad_params_loss_batch = jax.grad(loss_batch, argnums=0)


""" def init_mlp_params(sizes, key) """
def init_mlp_params(sizes, key):
    # sizes: list of layer sizes, e.g. [n_inputs, 10, 10, n_outputs]
    params = []
    keys = jax.random.split(key, len(sizes) - 1)  # osobny klucz dla każdej warstwy
    
    for i in range(len(sizes) - 1):
        n1 = sizes[i]
        n2 = sizes[i+1]
        k = keys[i]
        
        # Xavier / Glorot initialization dla tanh
        std = jnp.sqrt(2.0 / (n1 + n2))
        W = std * jax.random.normal(k, (n2, n1))
        b = jnp.zeros(n2)
        
        params.append((W, b))
    
    return params


# _____________________________________________________________
# Inicjalizacja i uczenie sieci MLP

layer_sizes = [n_inputs, 10, 10, n_outputs]
mlp_params = init_mlp_params(layer_sizes, jax.random.PRNGKey(2))
velocity = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in mlp_params]


def update_layer(pv, g, lr, mom):
    (W, b), (vW, vb) = pv
    dW, db = g
    vW_new = mom * vW - lr * dW
    vb_new = mom * vb - lr * db
    W_new = W + vW_new
    b_new = b + vb_new
    return (W_new, b_new), (vW_new, vb_new)

def update_params(params, x_batch, y_batch, lr, mom, velocity):
    grads = grad_params_loss_batch(params, x_batch, y_batch)

    new_params, new_velocity = jax.tree.map(
        lambda p, v, g: update_layer((p, v), g, lr, mom),
        params,
        velocity,
        grads
    )

    return new_params, new_velocity

# JIT całej funkcji
update_params_jit = jax.jit(update_params)

# Uczenie sieci
n_epochs = 5000
learning_rate = 0.01
momentum = 0.9
batch_size = 64

key = jax.random.PRNGKey(42)
for epoch in range(n_epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, n_train, (batch_size,), replace=False)
    X_batch = X_train[idx]
    Y_batch = Y_train[idx]
    mlp_params, velocity = update_params_jit(mlp_params, X_batch, Y_batch, learning_rate, momentum, velocity)

    if epoch % 100 == 0:
        loss = loss_val_full_jit(mlp_params)
        print(f"[{epoch}/{n_epochs}] Loss: {loss} \r", end='')