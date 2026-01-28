# %% ================================================================
# Control engineering: Phase portrait of a discrete-time system


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

cpu = jax.devices("cpu")[0]
jax.config.update("jax_default_device", cpu)

@jax.jit
def simulation(x0, k):
    # x(k+1) = f(x(k), u(k))
    def f(x, u):
        x1, x2 = x
        x1_new = 0.7 * x1 + 0.1 * x2 + 0.1 * u
        x2_new = 0.9 * x2 + 0.05 * u
        return jnp.array([x1_new, x2_new])

    # y(k) = g(x(k))
    def g(x):
        return 1.0 * x[0] + 0.1 * x[1]

    def step(carry, k):
        # carry: x(k)
        x = carry

        y = g(x)
        u = 0.0  # brak sterowania

        save = (y, x)
        x = f(x, u)
        carry = x
        return carry, save

    _, saved = jax.lax.scan(step, x0, k)
    y, x = saved
    return y, x

k = jnp.arange(0, 100)

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



