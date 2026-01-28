# %% ================================================================
# Control engineering: PID


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

cpu = jax.devices("cpu")[0]
jax.config.update("jax_default_device", cpu)

@jax.jit
def simulation(x0, k):
    kp = 1.0
    ki = 0.5

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
        # carry: x(k), u(k-1), e(k-1)
        x, u_prev, e_prev = carry

        y = g(x)
        y_zad = (k < 50) * 1.0 + (k >= 50) * -1.0
        e = y_zad - y
        du = kp * (e - e_prev) + ki * e
        du = jnp.clip(du, -0.2, 0.2)
        u = u_prev + du

        save = (y, y_zad, u)
        x = f(x, u)
        carry = (x, u, e)
        return carry, save

    def run(x0, k):
        u_prev = 0.0
        e_prev = 0.0
        _, saved = jax.lax.scan(step, (x0, u_prev, e_prev), k)
        return saved
    
    return run(x0, k)


k = jnp.arange(0, 150)
x0 = jnp.array([0.0, 0.0])
y, y_zad, u = simulation(x0, k)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(k, y, color='blue')
ax1.plot(k, y_zad, color='red', linestyle='--')
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2.plot(k, u, color='blue')
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()




