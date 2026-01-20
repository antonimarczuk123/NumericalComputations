# %% ===================================================================
# importy

import time
import jax.numpy as jnp
import jax.random as jrd
from jax import grad, vmap
from jax import jit
from jax.lax import scan
from jax.lax import fori_loop


# %% ===================================================================
# grad

def f(x,y):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    y1 = y[0]
    y2 = y[1]

    return jnp.sin(x1 * x3 - y1) - jnp.cos(y2 * x2)

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
# Specjanie nie robimy splita dla każdego rozmiaru, aby pokazać, że
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
# jit + generowanie losowych liczb w funkcji + funkcja jit w pętli

# Funkcja korzystająca z losowych liczb, musi otrzymywać klucz losowy jako argument
# i zwracać nowy klucz po użyciu. To kontrakt funkcjonalny JAXa.

# Pierwsze wywołanie jit_f będzie wolne, bo kompiluje funkcję.
# Kolejne wywołania będą szybkie, bo już skompilowane.


# Inicjalizacja klucza losowego
key = jrd.key(0)
x = jnp.linspace(0.0, 1.0, 1000)

# Definicja funkcji
def f(x, key, i):
    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    out = jnp.sin(x + w - i)
    return out, key

jit_f = jit(f)

# liczba iteracji pętli
N = 10_000 

# Wywołanie bez JIT

start = time.time()

for i in range(N):
    x, key = f(x, key, i)

end = time.time()
print(f"\nTime for {N} iterations: {end - start} seconds\n")

print(x[:10])

# Wywołanie z JIT

start = time.time()

for i in range(N):
    x, key = jit_f(x, key, i)
x.block_until_ready()

end = time.time()
print(f"\nTime for {N} iterations: {end - start} seconds\n")

print(x[:10])


# %% ===================================================================
# scan z jit na scan_body

# scan_body - wrapper dla scan. 
# scan można stosować bez jit, ale zastosowanie jit może przyspieszyć działanie.
# scan_body jest kompilowane tylko raz, a potem wywoływane wiele razy w ramach scan.

# carry:    to co przechodzi z kroku na krok (x, key) -> (x_next, key_next)
# i:        aktualna wartość z sekwencji wejściowej

@jit
def scan_body(carry, i):
    x, key = carry

    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    sum = jnp.sum(x)        # Obliczamy sumę elementów x jako wynik pośredni  

    return (x, key), sum

# Pomiar czasu

key_init = jrd.key(0)
x_init = jnp.linspace(0.0, 1.0, 1000)
iters = jnp.arange(200_000)

start = time.time()

(x_final, key_final), sums = scan(scan_body, (x_init, key_init), iters)

x_final.block_until_ready()
end = time.time()

print(sums[:10])        # wydruk pierwszych 10 sum
print(x_final[:10])     # wydruk pierwszych 10 x_final
print(f"\nTime for {len(iters)} iterations: {end - start} seconds\n")


# %% ===================================================================
# scan z jit na cały scan

# scan_body - wrapper dla scan. 
# carry:    to co przechodzi z kroku na krok (x, key) -> (x_next, key_next)
# i:        aktualna wartość z sekwencji wejściowej
def scan_body(carry, i):
    x, key = carry

    key, subkey = jrd.split(key)
    w = jrd.normal(subkey, x.shape)
    x = jnp.sin(x + w - i)

    sum = jnp.sum(x)        # Obliczamy sumę elementów x jako wynik pośredni  

    return (x, key), sum

# super_fast_loop - funkcja opakowująca cały scan w jit
# Dzięki temu cały scan jest skompilowany i działa bardzo szybko.
@jit
def super_fast_loop(x_init, key_init, iters):
    (x_final, key_final), sums = scan(scan_body, (x_init, key_init), iters)
    return (x_final, key_final), sums

# Pomiar czasu

key_init = jrd.key(0)
x_init = jnp.linspace(0.0, 1.0, 1000)
iters = jnp.arange(200_000)

start = time.time()

(x_final, key_final), sums = super_fast_loop(x_init, key_init, iters)

x_final.block_until_ready()
end = time.time()

print(sums[:10])        # wydruk pierwszych 10 sum
print(x_final[:10])     # wydruk pierwszych 10 x_final
print(f"\nTime for {len(iters)} iterations: {end - start} seconds\n")













