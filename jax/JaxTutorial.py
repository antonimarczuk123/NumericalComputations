# %% ===================================================================
# importy

import time

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap
from jax import jit
from jax import grad
from jax import value_and_grad
from jax import jacrev, jacfwd
from jax import hessian
from jax import jvp, vjp
from jax.tree_util import tree_map
from jax.lax import fori_loop
from jax.lax import scan
from jax.lax import while_loop
from jax.lax import cond
from jax.lax import switch
from jax.debug import print as jprint
from jax.experimental import io_callback
from jax import device_put


"""
construct  |  jit  |  grad
--------------------------
if         |   x   |   v
for        |   x   |   v
while      |   x   |   v
cond       |   v   |   v
while_loop |   v   |  fwd
fori_loop  |   v   |  fwd
scan       |   v   |   v
"""



# %% ===================================================================
# Praca z urządzeniami (CPU, GPU)

# --------------------
# Dostępne urządzenia
print("Urządzenia CPU:", jax.devices("cpu"))
print("Urządzenia GPU:", jax.devices("gpu"))

# --------------------
# Zapisywanie referencji do urządzeń
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

# --------------------
# Tworzenie tablic na CPU i GPU
x = jnp.ones(3, dtype=jnp.float32, device=cpu)
print("x.device: ", x.device)

y = jnp.ones(3, dtype=jnp.float32, device=gpu)
print("y.device: ", y.device)

# W JAX obliczenia podążają za danymi, więc operacje na tablicach
# będą wykonywane na urządzeniach, na których te tablice się znajdują.
# Dotyczy to również funkcji, również tych kompilowanych przez jit.

# --------------------
# Przenoszenie tablic między urządzeniami
gpu_x = device_put(x, device=gpu)
print("gpu_x.device: ", gpu_x.device)

cpu_y = device_put(y, device=cpu)
print("cpu_y.device: ", cpu_y.device)

# --------------------
# Ustawienie domyślnego urządzenia na GPU
jax.config.update("jax_default_device", gpu)

x = jnp.ones(3, dtype=jnp.float32)
print("x.device (default GPU): ", x.device)

# --------------------
# Ustawienie domyślnego urządzenia na CPU
jax.config.update("jax_default_device", cpu)

x = jnp.ones(3, dtype=jnp.float32)
print("x.device (default CPU): ", x.device)



# %% ===================================================================
# grad

def f(x,y):
    return jnp.sin(x[0] * x[2] - y[0]) - jnp.cos(y[1] * x[1])

Dxf = grad(f, 0) # Dxf(x,y)
Dyf = grad(f, 1) # Dyf(x,y)

x = jnp.array([0.5, 1.0, 1.5])
y = jnp.array([2.0, 3.0])

f_xy = f(x, y)
Dxf_xy = Dxf(x, y)
Dyf_xy = Dyf(x, y)

print(f_xy)
print(Dxf_xy)
print(Dyf_xy)



# %% ===================================================================
# vmap

def f(x,y):
    return jnp.sin(x[0] * x[2] - y[0]) - jnp.cos(y[1] * x[1])

Dxf = grad(f, 0) # Dxf(x,y)
Dyf = grad(f, 1) # Dyf(x,y)

vmap_f = vmap(f, in_axes=(0, 0))        # vmap_f(xBatch, yBatch)
vmap_Dxf = vmap(Dxf, in_axes=(0, 0))    # vmap_Dxf(xBatch, yBatch)
vmap_Dyf = vmap(Dyf, in_axes=(0, 0))    # vmap_Dyf(xBatch, yBatch)

xBatch = jnp.array([
    [0.5, 1.0, 1.5],
    [1.0, 2.0, 3.0],
    [1.5, 2.5, 3.5]
])

yBatch = jnp.array([
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

vmap_f_xy = vmap_f(xBatch, yBatch)
vmap_Dxf_xy = vmap_Dxf(xBatch, yBatch)
vmap_Dyf_xy = vmap_Dyf(xBatch, yBatch)

print(vmap_f_xy)
print()

print(vmap_Dxf_xy)
print()

print(vmap_Dyf_xy)
print()



# %% ===================================================================
# random

# JAX używa własnego generatora liczb losowych, który jest funkcjonalny - Threefry.
# Zawsze gdy generujemy losowe liczby, robimy split klucza!!!
# Nowy key trzymamy na przyszłość do kolejnych splitów, a subkey używamy do generowania liczb.

# Inicjalizacja klucza losowego
seed = round(time.time_ns() % 1e6)
key = jrd.key(seed)

# Generowanie liczb losowych z rozkładu normalnego
# W tym miejscu specjalnie nie robimy splita dla każdego rozmiaru, aby pokazać, że
# dostajemy te same liczby przy tym samym subkeyu. (Kolejny liczby sa generowane dla 
# tego samego subkeya, ale counter się zwiększa co jeden).
key, subkey = jrd.split(key)
x1 = jrd.normal(subkey, (2,))
x2 = jrd.normal(subkey, (4,))
x3 = jrd.normal(subkey, (6,))

print(x1)
print(x2)
print(x3)
print()

# Prawidłowe użycie splita
key, subkey = jrd.split(key)
x1 = jrd.normal(subkey, (2,))

key, subkey = jrd.split(key)
x2 = jrd.normal(subkey, (4,))

key, subkey = jrd.split(key)
x3 = jrd.normal(subkey, (6,))

print(x1)
print(x2)
print(x3)
print()

# Generowanie liczb losowych z rozkładu jednostajnego
key, subkey = jrd.split(key)
y = jrd.uniform(subkey, (6,), minval=0.0, maxval=1.0)
print(y)
print()

# Generowanie liczb całkowitych losowych
key, subkey = jrd.split(key)
z = jrd.randint(subkey, (6,), minval=0, maxval=10)
print(z)
print()

# Wybór losowych elementów z tablicy
arr = jnp.arange(start=0, stop=20, step=1)
print(arr)

key, subkey = jrd.split(key)
arr2 = jrd.choice(subkey, arr, shape=(5,), replace=False)
print(arr2)



# %% ===================================================================
# jit

def f(x,y):
    return jnp.sin(x[0] * x[2] - y[0]) - jnp.cos(y[1] * x[1])

Dxf = grad(f, 0) # Dxf(x,y)
Dyf = grad(f, 1) # Dyf(x,y)

jit_f = jit(f)
jit_Dxf = jit(Dxf)
jit_Dyf = jit(Dyf)

x = jnp.array([0.5, 1.0, 1.5])
y = jnp.array([2.0, 3.0])

jit_f_xy = jit_f(x, y)
jit_Dxf_xy = jit_Dxf(x, y)
jit_Dyf_xy = jit_Dyf(x, y)

print(jit_f_xy)
print(jit_Dxf_xy)
print(jit_Dyf_xy)
print()



# %% ===================================================================
# jit + vmap

def f(x,y):
    return jnp.sin(x[0] * x[2] - y[0]) - jnp.cos(y[1] * x[1])

Dxf = grad(f, 0) # Dxf(x,y)
Dyf = grad(f, 1) # Dyf(x,y)

vmap_f = vmap(f, in_axes=(0, 0))        # vmap_f(xBatch, yBatch)
vmap_Dxf = vmap(Dxf, in_axes=(0, 0))    # vmap_Dxf(xBatch, yBatch)
vmap_Dyf = vmap(Dyf, in_axes=(0, 0))    # vmap_Dyf(xBatch, yBatch)

jit_vmap_f = jit(vmap_f)
jit_vmap_Dxf = jit(vmap_Dxf)
jit_vmap_Dyf = jit(vmap_Dyf)

xBatch = jnp.array([
    [0.5, 1.0, 1.5],
    [1.0, 2.0, 3.0],
    [1.5, 2.5, 3.5]
])

yBatch = jnp.array([
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

jit_vmap_f_xy = jit_vmap_f(xBatch, yBatch)
jit_vmap_Dxf_xy = jit_vmap_Dxf(xBatch, yBatch)
jit_vmap_Dyf_xy = jit_vmap_Dyf(xBatch, yBatch)

print(jit_vmap_f_xy)
print()
print(jit_vmap_Dxf_xy)
print()
print(jit_vmap_Dyf_xy)
print()



# %% ===================================================================
# for loop z jit na ciało pętli

# Definicja funkcji
def for_body(x, key, i):
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)
    return x, key

jit_for_body = jit(for_body)

N = 10_000 
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)

for i in range(N):
    x, key = jit_for_body(x, key, i)

x.block_until_ready()

print(x[:10])



# %% ==================================================================
# fori_loop z jit na ciało pętli

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)
    return (x, key)

jit_for_body = jit(for_body)

N = 10_000
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)

x, key = fori_loop(0, N, jit_for_body, (x, key))
x.block_until_ready()

print(x[:10])


"""
The semantics of fori_loop are given by this Python implementation:

def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val
"""



# %% ===================================================================
# fori_loop z jit na cały fori_loop

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)
    return (x, key)

@jit
def super_fast_loop(lower, upper, x_init, key_init):
    x, key = fori_loop(lower, upper, for_body, (x_init, key_init))
    return x, key

N = 10_000
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)
x, key = super_fast_loop(0, N, x, key)
x.block_until_ready()

print(x[:10])



# %% ===================================================================
# scan z jit na ciało pętli

def scan_body(carry, z):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - z)
    partial_result = jnp.sum(x) # Obliczamy sumę elementów x jako wynik pośredni  
    return (x, key), partial_result

jit_scan_body = jit(scan_body)

key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)

zs = jnp.linspace(0.0, 1.0, 10_000)

(x, key), partial_results = scan(jit_scan_body, (x, key), zs)
x.block_until_ready()

print(partial_results[:10])     # wydruk pierwszych 10 sum
print(x[:10])                   # wydruk pierwszych 10 x


"""
The semantics of scan are given by this Python implementation:

def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
"""



# %% ===================================================================
# scan z jit na cały scan


def scan_body(carry, z):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - z)
    partial_result = jnp.sum(x) # Obliczamy sumę elementów x jako wynik pośredni  
    return (x, key), partial_result

@jit
def super_fast_scan(x_init, key_init, zs):
    (x, key), partial_results = scan(scan_body, (x_init, key_init), zs)
    return (x, key), partial_results

key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)

zs = jnp.linspace(0.0, 1.0, 10_000)

(x, key), partial_results = super_fast_scan(x, key, zs)
x.block_until_ready()

print(partial_results[:10])     # wydruk pierwszych 10 sum
print(x[:10])                   # wydruk pierwszych 10 x



# %% ===================================================================
# cond

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    x = cond(i % 10_000 == 0,
        lambda x, w: x - w,
        lambda x, w: x + w, 
        x, w)

    return (x, key)

@jit
def super_fast_loop(lower, upper, x_init, key_init):
    x, key = fori_loop(lower, upper, for_body, (x_init, key_init))
    return x, key

N = 20_000
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)
x, key = super_fast_loop(0, N, x, key)
x.block_until_ready()

print("Loop finished.")
print(x[:10])


"""
The semantics of cond are given by this Python implementation:

def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)
"""



# %% ==================================================================
# switch

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    x = switch(i % 3, [
        lambda x, w: x + w,
        lambda x, w: x - w,
        lambda x, w: x * w
        ], x, w )

    return (x, key)

@jit
def super_fast_loop(lower, upper, x_init, key_init):
    x, key = fori_loop(lower, upper, for_body, (x_init, key_init))
    return x, key

N = 20_000
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)
x, key = super_fast_loop(0, N, x, key)
x.block_until_ready()

print("Loop finished.")
print(x[:10])


"""
The semantics of switch are given by this Python implementation:

def switch(index, branches, *operands):
    index = clamp(0, index, len(branches) - 1)
    return branches[index](*operands)
"""



# %% ===================================================================
# cond + jprint do monitorowania przebiegu obliczeń

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    cond(i % 10_000 == 0,
        lambda i: jprint("Iteration {}", i),
        lambda i: None, 
        i)

    return (x, key)

@jit
def super_fast_loop(lower, upper, x_init, key_init):
    x, key = fori_loop(lower, upper, for_body, (x_init, key_init))
    return x, key

N = 200_000
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)
x, key = super_fast_loop(0, N, x, key)
x.block_until_ready()

print("Loop finished.")
print(x[:10])



# %% ===================================================================
# cond + io_callback do monitorowania przebiegu obliczeń

N = 200_000

def progress_print(i):
    progress = (i / N) * 100
    print(f"\rIteration {progress:.2f}%", end='')

def for_body(i, carry):
    x, key = carry
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    cond(i % 10_000 == 0,
        lambda i: io_callback(progress_print, None, i),
        lambda i: None, 
        i)

    return (x, key)

@jit
def super_fast_loop(lower, upper, x_init, key_init):
    x, key = fori_loop(lower, upper, for_body, (x_init, key_init))
    return x, key

key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)
x, key = super_fast_loop(0, N, x, key)
x.block_until_ready()

print("\nLoop finished.")
print(x[:10])



# %% ===================================================================
# value_and_grad

def f(x,y):
    return jnp.sin(x[0] * x[2] - y[0]) - jnp.cos(y[1] * x[1])

val_and_Dx_f = value_and_grad(f, 0) # val_and_Dxf(x,y)
val_and_Dy_f = value_and_grad(f, 1) # val_and_Dyf(x,y)

x = jnp.array([0.5, 1.0, 1.5])
y = jnp.array([2.0, 3.0])

f_xy, Dxf_xy = val_and_Dx_f(x, y)
print(f_xy)
print(Dxf_xy)

f_xy, Dyf_xy = val_and_Dy_f(x, y)
print(f_xy)
print(Dyf_xy)



# %% ==================================================================
# while_loop z jit na całą pętlę

def cond_fun(carry):
    x, key, i = carry
    s = jnp.sum(x)

    cond_i = (i < 100_000)
    cond_s = (s < 1000.0)

    # operatory bitowe: &, |, ~
    return cond_i & cond_s

def body_fun(carry):
    x, key, i = carry
    key, subkey = jrd.split(key)
    w = jrd.uniform(subkey, x.shape, minval=0.0, maxval=2e-5)
    x = x + w
    i += 1
    s = jnp.sum(x)

    cond(i % 10000 == 0,
        lambda args: jprint("Iteracja: {}, Suma: {}", args[0], args[1]),
        lambda args: None,
        (i, s))

    return (x, key, i)

@jit
def super_fast_while_loop(x_init, key_init):
    init_carry = (x_init, key_init, 0)
    x, key, i = while_loop(cond_fun, body_fun, init_carry)
    return x, key, i

x = jnp.linspace(0.0, 1.0, 1000)
key = jrd.key(0)
x, key, i = super_fast_while_loop(x, key)
x.block_until_ready()

print("While loop finished at iteration:", i)
print(jnp.sum(x))


"""
The semantics of while_loop are given by this Python implementation:

def while_loop(cond_fun, body_fun, init_carry):
    carry = init_carry
    while cond_fun(carry):
        carry = body_fun(carry)
    return carry
"""



# %% ==================================================================
# PyTree

# PyTree to zagnieżdżona struktura danych, której liśćmi są tablice JAX (jnp.ndarray).
# JAX operuje na PyTrees zachowując ich strukturę, np. wykonując grad dostajemy PyTree z 
# pochodnymi o tej samej strukturze co struktura argumentu wejściowego funkcji.

def f(x):
    return jnp.sum(x['a'] ** 2) + jnp.sum(x['b']['c'] ** 3) + jnp.sum(x['b']['d'] ** 4)

Dxf = grad(f)  # Dxf(x)

x = {
    'a': jnp.array([1.0, 2.0, 3.0]),
    'b': {
        'c': jnp.array([4.0, 5.0]),
        'd': jnp.array([6.0])
    }
}

f_x = f(x)
Dxf_x = Dxf(x)

print(f_x)
print()

print(Dxf_x)
print()



# %% ===================================================================
# tree_map

# tree_map pozwala na zastosowanie funkcji do odpowiadających sobie liści.
# Dzięki temu możamy np. łatwo aktualizować wagi w uczeniu sieci neuronowych.

params = {
    'a': jnp.array([1.0, 2.0, 3.0]),
    'b': {
        'c': jnp.array([4.0, 5.0]),
        'd': jnp.array([6.0])
    }
}

grads = {
    'a': jnp.array([10.0, 20.0, 30.0]),
    'b': {
        'c': jnp.array([40.0, 50.0]),
        'd': jnp.array([60.0])
    }
}

lr = 0.01

new_params = tree_map(lambda p, g: p - lr * g, params, grads)

print(new_params)



# %% =================================================================
# hessian

# x -> f(x)
def f(x):
    return jnp.sin(x[0] * x[2]) - jnp.cos(x[1])

# x -> Hxf(x)
Hf = hessian(f)

x = jnp.array([0.5, 1.0, 1.5])
Hf_x = Hf(x)

print(Hf_x)



# %% ===================================================================
# Przydatne i proste liczenie hessjanu razy wektor.
# Hxf(x) @ v = Dx ( Dxf(x) dot v ) 

# x -> f(x)
def f(x):
    return jnp.sin(x[0] * x[2]) - jnp.cos(x[1])

# x -> Dxf(x)
Dxf = grad(f)

# (x, v) -> Dxf(x) dot v
Dxf_dot_v = lambda x, v: jnp.vdot(Dxf(x), v)

# (x, v) -> Hxf(x) @ v
Hfv = grad(Dxf_dot_v, argnums=0)

x = jnp.array([0.5, 1.0, 1.5])
v = jnp.array([1.0, 0.0, -1.0])
Hfv_xv = Hfv(x, v)

print(Hfv_xv)



# %% ===================================================================
# Przydatne i proste liczenie zakrzywienia w kierunku v.
# v.T @ Hxf(x) @ v = v dot Dx ( Dxf(x) dot v )

# x -> f(x)
f = lambda x: jnp.sin(x[0] * x[2]) - jnp.cos(x[1])

# x -> Dxf(x)
Dxf = grad(f)

# (x, v) -> Dxf(x) dot v
Dxf_dot_v = lambda x, v: jnp.vdot(Dxf(x), v)

# (x, v) -> Hxf(x) @ v
Hfv = grad(Dxf_dot_v, argnums=0)

# (x, v) -> v.T @ Hxf(x) @ v
vHfv = lambda x, v: jnp.vdot(v, Hfv(x, v))

x = jnp.array([0.5, 1.0, 1.5])
v = jnp.array([1.0, 0.0, -1.0])

vHfv_xv = vHfv(x, v)
print(vHfv_xv)



# %% ===================================================================
# jacfwd, jacrev

"""These two functions compute the same values (up to machine numerics), but differ in their implementation: jacfwd uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices (more outputs than inputs), while jacrev uses reverse-mode, which is more efficient for “wide” Jacobian matrices (more inputs than outputs). For matrices that are near-square, jacfwd probably has an edge over jacrev"""

def f(x):
    y0 = x[0] ** 2 + x[4]
    y1 = jnp.sin(x[1]) + x[2]
    y2 = jnp.exp(x[3]) + jnp.log(x[2] + 1.0)
    return jnp.array([y0, y1, y2])

Jf_fwd = jacfwd(f)  # Jf_fwd(x)
Jf_rev = jacrev(f)  # Jf_rev(x)

x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])

Jf_fwd_x = Jf_fwd(x)
Jf_rev_x = Jf_rev(x)

print(Jf_fwd_x)
print()

print(Jf_rev_x)
print()



# %% ===================================================================
# Hessian via jacfwd and jacrev

"""To implement hessian, we could have used jacfwd(jacrev(f)) or jacrev(jacfwd(f)) or any other composition of the two. But forward-over-reverse is typically the most efficient. That is because in the inner Jacobian computation we are often differentiating a function wide Jacobian (maybe like a loss function f: R^n -> R), while in the outer Jacobian computation we are differentiating a function with a square Jacobian (since Df: R^n -> R^n ), which is where forward-mode wins out."""

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]

def Hess(f):
    return jacfwd(jacrev(f))

def f(x):
    return jnp.sin(x[0] * x[2]) - jnp.cos(x[1])

Hf = Hess(f)  # Hf(x)

x = jnp.array([0.5, 1.0, 1.5], device=gpu)
Hf_x = Hf(x)

print(Hf_x)
print(Hf_x.device)



# %% ===================================================================
# jvp

def f(x):
    y0 = x[0] ** 2 + x[4]
    y1 = jnp.sin(x[1]) + x[2]
    y2 = jnp.exp(x[3]) + jnp.log(x[2] + 1.0)
    return jnp.array([y0, y1, y2])

x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
v = jnp.array([1.0, 0.0, -1.0, 0.5, 2.0])

# jvp: f, (x,), (v,) -> f(x), Dxf(x) @ v
f_val, Dfv_val = jvp(f, (x,), (v,))

print(f_val)
print(Dfv_val)
print()

# -------------------------------------------

def f(x,y):
    z0 = x[0] ** 2 + x[4] - y[0]
    z1 = jnp.sin(x[1]) + x[2] + y[1]
    z2 = jnp.exp(x[3]) + jnp.log(x[2] + 1.0)
    return jnp.array([z0, z1, z2])

x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
y = jnp.array([0.1, 0.2])

vx = jnp.array([1.0, 0.0, -1.0, 0.5, 2.0])
vy = jnp.array([0.5, -0.5])

# jvp: f, (x, y), (vx, vy) -> f(x,y), Dxf(x,y) @ vx + Dyf(x,y) @ vy
f_val, Dfv_val = jvp(f, (x, y), (vx, vy))

print(f_val)
print(Dfv_val)
print()

"""If we apply a JVP to a one-hot tangent vector, it reveals one column of the Jacobian matrix, corresponding to the nonzero entry we fed in. So we can build a full Jacobian one column at a time, and to get each column costs about the same as one function evaluation. That will be efficient for functions with “tall” Jacobians, but inefficient for “wide” Jacobians."""



# %% ===================================================================
# vjp

def f(x):
    y0 = x[0] ** 2 + x[4]
    y1 = jnp.sin(x[1]) + x[2]
    y2 = jnp.exp(x[3]) + jnp.log(x[2] + 1.0)
    return jnp.array([y0, y1, y2])

x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])

# vjp: f, x -> f(x), vDf_fun
f_val, vDf_fun = vjp(f, x)

# Obliczanie kolumn Jacobiana poprzez mnożenie przez wektory jednostkowe.
# vDf_fun zwraca krotkę, w razie jednej zmiennej wejściowej mamy jeden element krotki.
J0 = vDf_fun(jnp.array([1.0, 0.0, 0.0]))  # Pierwsza kolumna Jacobiana
J1 = vDf_fun(jnp.array([0.0, 1.0, 0.0]))  # Druga kolumna Jacobiana
J2 = vDf_fun(jnp.array([0.0, 0.0, 1.0]))  # Trzecia kolumna Jacobiana

print(f_val)
print(J0)
print(J1)
print(J2)
print()

# -------------------------------------------

def f(x,y):
    z0 = x[0] ** 2 + x[4] - y[0]
    z1 = jnp.sin(x[1]) + x[2] + y[1]
    z2 = jnp.exp(x[3]) + jnp.log(x[2] + 1.0)
    return jnp.array([z0, z1, z2])

x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
y = jnp.array([0.1, 0.2])

# vjp: f, (x, y) -> f(x,y), vDf_fun
f_val, vDf_fun = vjp(f, x, y)

# Obliczanie kolumn Jacobiana poprzez mnożenie przez wektory jednostkowe.
# vDf_fun zwraca krotkę, w razie wielu zmiennych wejściowych mamy wiele elementów krotki
# odpowiadających poszczególnym zmiennym wejściowym.
J0 = vDf_fun(jnp.array([1.0, 0.0, 0.0]))  # Pierwsza kolumna Jacobiana
J1 = vDf_fun(jnp.array([0.0, 1.0, 0.0]))  # Druga kolumna Jacobiana
J2 = vDf_fun(jnp.array([0.0, 0.0, 1.0]))  # Trzecia kolumna Jacobiana

J0x, J0y = J0
J1x, J1y = J1
J2x, J2y = J2

print(f_val)
print(J0)
print(J1)
print(J2)
print()

"""This is great because it lets us build Jacobian matrices one row at a time. if we want the gradient of a function f: R^n -> R, we can do it in just one call. That's how 'grad' is efficient for gradient-based optimization, even for objectives like neural network training loss functions on millions or billions of parameters."""



