# ==========================================================
# Regulacja PID obiektu nieliniowego z małym opóźnieniem

# Przyjęty okres próbkowania regulatora: Ts = 125sek.
# Symulacja działa z krokiem h = Ts/20 = 6.25sek i wykorzystuje metodę RK4.
# Opóźnienie sterowania: 2 okres próbkowania = 250sek.
# Wykorzystano kompilację numba njit dla przyspieszenia długich symulacji.

# Dzięki zastosowaniu adaptacyjnych nastawów regulatora PID mamy dobrą jakość regulacji
# w całym szerokim zakresie pracy mimo, że obiekt jest nieliniowy.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# dx1/dt(t) = ( u(t-250) + z(t) -23 sqrt(x1(t)) ) / ( 0.7 x1(t) )
# dx2/dt(t) = ( 23 sqrt(x1(t)) - 30 sqrt(x2(t)) ) / ( 1.35 x2(t)^2 )
# y(t) = x2(t)

# =========================================================
# Kod symulacji

# TODO: zmienić implementację opóźnienia na całkowitą wielokrotność kroku symulacji.

@njit
def obj(u_del, z, x):
    x1, x2 = x
    x1 = np.maximum(1e-3, x1)
    x2 = np.maximum(1e-3, x2)
    dx1_dt = (u_del + z - 23.0 * np.sqrt(x1)) / (0.7 * x1)
    dx2_dt = (23.0 * np.sqrt(x1) - 30.0 * np.sqrt(x2)) / (1.35 * x2**2)
    return np.array([dx1_dt, dx2_dt])

@njit
def controller(e, e_prev1, e_prev2, Ts, uu, y):
    # PID controller parameters
    kd = 0.0 # nie używamy członu różniczkującego
    ki = 0.005 # wspólny współczynnik całkujący
    # Adaptacyjnie dobierane wzmocnienie kp ze względu na nieliniowość obiektu
    kp = 2.0 + 3.0 * (y - 30.0) / 20.0

    du = kp * (e - e_prev1) + ki * Ts / 2.0 * (e + e_prev1) + kd / Ts * (e - 2.0 * e_prev1 + e_prev2)
    du = np.maximum(-2.0, np.minimum(2.0, du)) # ograniczenie przyrostu sterowania
    uu = uu + du
    uu = np.maximum(0.0, np.minimum(400.0, uu)) # ograniczenie sygnału sterującego
    return uu

@njit
def simulate():
    x0 = np.array([20.0, 20.0]) # initial state
    t0 = 0.0 # start time
    tf = 360_000.0 # end time

    Ts = 125.0 # okres próbkowania regulatora
    
    reg_sim_factor = 100 # ile razy szybsza symulacja niż okres próbkowania
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
    r[(0 <= t) & (t < 20_000)] = 15
    r[(20_000 <= t) & (t < 80_000)] = 100
    r[(80_000 <= t) & (t < 150_000)] = 50
    r[(150_000 <= t) & (t < 200_000)] = 80
    r[(200_000 <= t) & (t < 300_000)] = 200
    r[(300_000 <= t)] = 50

    # zakłócenie
    z[(0 <= t) & (t < 10_000)] = 100.0
    z[(10_000 <= t) & (t < 50_000)] = 50.0
    z[(50_000 <= t) & (t < 120_000)] = 150.0
    z[(120_000 <= t) & (t < 170_000)] = 50.0
    z[(170_000 <= t) & (t < 260_000)] = 100.0
    z[(260_000 <= t)] = 200.0

    e = 0.0
    e_prev1 = 0.0
    e_prev2 = 0.0
    uu = 0.0
    x = x0

    for k in range(n_steps - 1):
        if k % 500_000 == 0:
            print("Simulating step", k, "/", n_steps - 1)

        # controller wyznacza sterowanie co Ts
        if k % reg_sim_factor == 0:
            e = r[k] - y[k]
            uu = controller(e, e_prev1, e_prev2, Ts, uu, y[k])
            e_prev2 = e_prev1
            e_prev1 = e

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

# =========================================================
# Uruchomienie symulacji

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


