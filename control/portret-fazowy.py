# %% ================================================================
# importy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

@jax.jit
def simulation(x0, k):

    A = jnp.array([[0.7, 0.1], [0.0, 0.9]])
    B = jnp.array([[0.1], [0.05]])
    C = jnp.array([1.0, 0.1])

    # x in R^2, u in R^1, y in R^1
    # x(k+1) = f(x(k), u(k))
    # y(k+1) = g(x(k+1))
    def obj(x, u):
        x_new = A @ x + B.squeeze() * u
        y_new = C @ x_new
        return y_new, x_new

    def step(carry, k):
        x_prev = carry
        y_new, x_new = obj(x_prev, 0.0)
        carry = x_new
        save = (y_new, x_new)
        return carry, save

    def run(x0, k):
        _, saved = jax.lax.scan(step, x0, k)
        y, x = saved
        return y, x
    
    return run(x0, k)

k = jnp.arange(0, 70)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for x10 in jnp.linspace(-3, 3, 10):
    for x20 in jnp.linspace(-3, 3, 10):
        x0 = jnp.array([x10, x20])
        y, x = simulation(x0, k)
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



