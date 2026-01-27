# %% ================================================================
# importy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# x in R^2, u in R^1, y in R^1
# x(k+1) = f(x(k), u(k))
# y(k+1) = g(x(k+1))
def obj(x, u):
    x1, x2 = x
    x1_new = 0.7 * x1 + 0.1 * x2 + 0.1 * u
    x2_new = 0.9 * x2 + 0.05 * u
    x_new = jnp.array([x1_new, x2_new])
    y_new = x1_new + 0.1 * x2_new
    return y_new, x_new

def step(carry, k):
    x_prev = carry
    y_new, x_new = obj(x_prev, 0.0)
    carry = x_new
    save = (y_new, x_new)
    return carry, save

@jax.jit
def simulate(x0, K):
    _, saved = jax.lax.scan(step, x0, K)
    y, x = saved
    return y, x

k = jnp.arange(0, 70)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for x10 in jnp.linspace(-3, 3, 10):
    for x20 in jnp.linspace(-3, 3, 10):
        x0 = jnp.array([x10, x20])
        y, x = simulate(x0, k)
        ax1.plot(x[:, 0], x[:, 1])
        ax2.plot(k, y)

ax1.set_title('State trajectory')
ax1.set_xlabel('State x1')
ax1.set_ylabel('State x2')
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2.set_title('Output y over time')
ax2.set_xlabel('Time step k')
ax2.set_ylabel('Output y')
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# fig = plt.figure()

# ax1 = fig.add_subplot(211)
# ax1.plot(K, yzad, linestyle='--', color='orange')
# ax1.plot(K, y, linestyle='-', color='blue')
# ax1.set_title('Output y')
# ax1.set_xlabel('Time step k')
# ax1.set_ylabel('y')
# ax1.minorticks_on()
# ax1.grid(True, which='major', linestyle='-')
# ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

# ax2 = fig.add_subplot(212)
# ax2.plot(K, u, linestyle='-', color='blue')
# ax2.set_title('Input u')
# ax2.set_xlabel('Time step k')
# ax2.set_ylabel('u')
# ax2.minorticks_on()
# ax2.grid(True, which='major', linestyle='-')
# ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()





