# %% ================================================================

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def simulation(steps, x0):
    # x(k+1) = f(x(k), u(k))
    def f(x, u):
        x1, x2 = x
        x1_new = 0.7 * x1 + 0.1 * x2 + 0.1 * u
        x2_new = 0.9 * x2 + 0.05 * u
        return (x1_new, x2_new)

    # y(k) = g(x(k))
    def g(x):
        return 1.0 * x[0] + 0.1 * x[1]
    
    # Alokacja tablic na wyniki
    y_out = np.zeros(steps)
    yzad_out = np.zeros(steps)
    u_out = np.zeros(steps)

    # Parametry regulatora PI
    kp = 1.0
    ki = 0.5
    
    # Inicjalizacja stanu
    x = x0
    u_prev = 0.0
    e_prev = 0.0
    
    for k in range(steps):
        # Odczyt aktualnego wyjścia
        y = g(x)

        # regulacja
        y_zad = (k < 50) * 1.0 + (k >= 50) * -1.0
        e = y_zad - y
        du = kp * (e - e_prev) + ki * e
        du = min(max(du, -0.2), 0.2)
        u = u_prev + du
        
        # Zapis wyników
        y_out[k] = y
        yzad_out[k] = y_zad
        u_out[k] = u
        
        # Aktualizacja obiektu (x(k) -> x(k+1))
        x = f(x, u)
        e_prev = e
        u_prev = u

    return y_out, yzad_out, u_out

# Parametry symulacji
N = 150
x_init = (0.0, 0.0)
y, y_zad, u = simulation(N, x_init)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(y, color='blue')
ax1.plot(y_zad, color='red', linestyle='--')
ax1.minorticks_on()
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--', alpha=0.5)

ax2.plot(u, color='blue')
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='-')
ax2.grid(True, which='minor', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



