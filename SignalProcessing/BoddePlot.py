# %% Importy

import numpy as np
import matplotlib.pyplot as plt


# %% Wykres Bodego


omega = np.logspace(-1, 2, 1000)  # od 0.1 do 100

num = 7 + 8j*omega - omega**2 + 1j*omega**2
den = 7 - omega**2 + omega*1j - 6j*omega**3

H = num/den

mag = 20 * np.log10(np.abs(H))      # w dB
phase = np.angle(H, deg=True)       # w stopniach

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.semilogx(omega, mag)
ax1.set_ylabel("Wzmocnienie [dB]")
ax1.grid(True, which="both")

ax2.semilogx(omega, phase)
ax2.set_xlabel("ω [rad/s]")
ax2.set_ylabel("Faza [°]")
ax2.grid(True, which="both")

plt.tight_layout()
plt.show()

