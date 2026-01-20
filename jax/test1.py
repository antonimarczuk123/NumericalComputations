import time
import jax
import jax.numpy as jnp

def inner_loop(i):
    x = jnp.arange(1000)
    y = i + jnp.sin(x) + jnp.cos(x)
    return jnp.sum(y ** 2) + i

inner_loop_jit = jax.jit(inner_loop)

# Pętla bez JIT
inner_loop(0).block_until_ready()
start = time.time()
for i in range(1, 10000):
    inner_loop(i).block_until_ready()
end = time.time()
print(f"Without JIT: {end - start:.4f} seconds")

# Pętla z JIT
inner_loop_jit(0).block_until_ready()
start = time.time()
for i in range(1, 10000):
    inner_loop_jit(i).block_until_ready()
end = time.time()
print(f"With JIT: {end - start:.4f} seconds")
