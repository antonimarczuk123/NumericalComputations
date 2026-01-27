# %% ================================================================
# importy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# %% ================================================================
# Obj

# y(k) = f(Yk, Uk, k)
# Yk = [y(k-1), y(k-2)]
# Uk = [u(k-1), u(k-2)]
def obj(Yk, Uk, k):
    return jnp.sqrt(jnp.maximum(0.0, 0.9*Yk[0] - 0.2*Yk[1] + 0.1*Uk[0])) + 0.05*jnp.power(Uk[1], 2)


# %% ================================================================
# symulacja

# Yk = [y(k-1), y(k-2)]
# Uk = [u(k-1), u(k-2)]
# ek_prev = e(k-1)
# ek_int = sum_{i=0}^{k-1} e(i)
def step(carry, k):
    Yk, Uk, ek_prev, ek_int = carry

    yk = obj(Yk, Uk, k)
    yk_zad = y_zad(k)

    ek = yk_zad - yk
    ek_int = ek_int + ek
    # ek_int = jnp.clip(ek_int + ek, -100.0, 100.0)

    uk = controller(ek, ek_prev, ek_int)

    Y_new = jnp.roll(Yk, 1)
    Y_new = Y_new.at[0].set(yk)
    U_new = jnp.roll(Uk, 1)
    U_new = U_new.at[0].set(uk)

    carry = (Y_new, U_new, ek, ek_int)
    save = (yk, yk_zad, uk)
    return carry, save

def controller(ek, ek_prev, ek_int):
    Kp = 0.001
    Ki = 0.005
    Kd = 0.0
    uk = Kp * ek + Ki * ek_int + Kd * (ek - ek_prev)
    return uk

def y_zad(k):
    yk_zad = jnp.where((k > 20) & (k < 700), 10, 0.0)
    return yk_zad

@jax.jit
def simulate(Y0, U0, K):
    _, qqq = jax.lax.scan(step, (Y0, U0, 0.0, 0.0), K)
    y, yzad, u = qqq
    return y, yzad, u


K = jnp.arange(0, 5000)
Y0 = jnp.array([0.0, 0.0])
U0 = jnp.array([0.0, 0.0])

y, yzad, u = simulate(Y0, U0, K)

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(K, yzad, linestyle='--', color='orange')
ax1.plot(K, y, linestyle='-', color='blue')
ax1.set_title('Output y')
ax1.set_xlabel('Time step k')
ax1.set_ylabel('y')
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2 = fig.add_subplot(212)
ax2.plot(K, u, linestyle='-', color='blue')
ax2.set_title('Input u')
ax2.set_xlabel('Time step k')
ax2.set_ylabel('u')
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()





