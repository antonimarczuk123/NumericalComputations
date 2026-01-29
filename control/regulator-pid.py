# ==========================================================
# Regulacja PID obiektu nieliniowego z opóźnieniem

# Przyjęty okres próbkowania regulatora: Ts = 125sek.
# Symulacja działa z krokiem h = Ts/20 = 6.25sek i wykorzystuje metodę RK4.
# Opóźnienie sterowania: 2 okres próbkowania = 250sek.
# Wykorzystano kompilację numba njit dla przyspieszenia długich symulacji.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# dx1/dt(t) = ( u(t-250) + z(t) -23 sqrt(x1(t)) ) / ( 0.7 x1(t) )
# dx2/dt(t) = ( 23 sqrt(x1(t)) - 30 sqrt(x2(t)) ) / ( 1.35 x2(t)^2 )
# y(t) = x2(t)

@njit
def obj(u_del, z, x):
    x1, x2 = x
    x1 = np.maximum(1e-3, x1)
    x2 = np.maximum(1e-3, x2)
    dx1_dt = (u_del + z - 23.0 * np.sqrt(x1)) / (0.7 * x1)
    dx2_dt = (23.0 * np.sqrt(x1) - 30.0 * np.sqrt(x2)) / (1.35 * x2**2)
    return np.array([dx1_dt, dx2_dt])

@njit
def simulate():
    x0 = np.array([50.0, 50.0]) # initial state
    t0 = 0.0 # start time
    tf = 130_000.0 # end time

    # PID controller parameters
    kp = 7.0
    ki = 0.01
    kd = 0

    Ts = 125.0 # okres próbkowania regulatora
    
    reg_sim_factor = 20 # ile razy szybsza symulacja niż okres próbkowania
    h = Ts / reg_sim_factor # krok symulacji
    n_steps = int((tf - t0) / h) + 1 # liczba kroków symulacji

    delay = 2.0 * Ts # opóźnienie sterowania - czas
    delay_steps = int(delay / h) # opóźnienie sterowania - ile kroków symulacji wstecz

    t = (t0 + np.arange(n_steps) * h).astype(np.float64)
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    r = np.zeros_like(t)
    z = np.zeros_like(t)

    # początkowe wyjście
    y[0] = x0[1]

    # wartość zadana
    r[(0 <= t) & (t <30_000)] = 50
    r[(30_000 <= t) & (t < 100_000)] = 100
    r[(100_000 <= t)] = 70

    # zakłócenie
    z[(0 <= t) & (t < 15_000)] = 100
    z[(15_000 <= t) & (t < 70_000)] = 50
    z[(70_000 <= t)] = 100

    e = np.zeros(3)
    uu = 0.0
    x = x0

    for k in range(n_steps - 1):
        if k % 5000 == 0:
            print("Simulating step", k, "/", n_steps - 1)

        # controller wyznacza sterowanie co Ts
        if k % reg_sim_factor == 0:
            e[0] = r[k] - y[k]
            du = kp * (e[0] - e[1]) + ki * Ts / 2.0 * (e[0] + e[1]) + kd / Ts * (e[0] - 2.0 * e[1] + e[2])
            du = np.maximum(-2.0, np.minimum(2.0, du)) # ograniczenie przyrostu sterowania
            uu = uu + du
            uu = np.maximum(0.0, np.minimum(400.0, uu)) # ograniczenie sygnału sterującego
            e[2] = e[1]
            e[1] = e[0]

        # zero-order hold trzymamy wartość sterowania do następnej aktualizacji regulatora
        u[k] = uu

        # krok symulacji - RK4 z krokiem h
        uu_delay = u[k-delay_steps] if k >= delay_steps else 0.0
        zz = z[k]
        k1 = obj(uu_delay, zz, x)
        k2 = obj(uu_delay, zz, x + h/2.0 * k1)
        k3 = obj(uu_delay, zz, x + h/2.0 * k2)
        k4 = obj(uu_delay, zz, x + h * k3)
        x = x + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        x = np.maximum(1e-3, x) # zabezpieczenie przed ujemnymi stanami
        y[k+1] = x[1]

    print("\rSimulation completed.")
    return t[:-1], y[:-1], u[:-1], r[:-1], z[:-1]

t, y, u, r, z = simulate()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(t, y, label='y(t)')
ax1.plot(t, r, '--', label='r(t)')
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)
ax1.legend()

ax2.step(t, u, label='u(t)', color='orange')
ax2.plot(t, z, '--', label='z(t)', color='green')
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)
ax2.legend()

plt.show()


