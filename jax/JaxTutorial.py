# %% ===================================================================
# importy

import time

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import grad, vmap
from jax import jit
from jax.lax import scan
from jax.lax import fori_loop
from jax import device_put



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












