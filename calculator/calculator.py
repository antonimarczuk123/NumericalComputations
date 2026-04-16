# %% ==========================================================================

import numpy as np
import matplotlib.pyplot as plt


# %% ==========================================================================
# AC circuit

U = 220
Z = 15 + 47*1j
I = U/Z

print()
print(f"I = {I:f}")
print(f"abs(I) = {np.abs(I):e}")
print(f"angle(I) = {np.angle(I, deg=True):f} [deg]")


# %% ==========================================================================
# Low-pass filter

R = 1e3
C = 1e-6

w = np.logspace(0, 6, 1000)

Lowpass = 1/(1 + 1j*w*R*C)

fig1 = plt.figure()

axMag = fig1.add_subplot(211)
axMag.loglog(w, np.abs(Lowpass))
axMag.set_ylabel('Magnitude')
axMag.grid(True, which='major', linestyle='-')
axMag.grid(True, which='minor', linestyle='--', alpha=0.5)

axPhase = fig1.add_subplot(212, sharex=axMag)
axPhase.semilogx(w, np.angle(Lowpass, deg=True))
axPhase.set_xlabel('Omega [rad/s]')
axPhase.set_ylabel('Phase [deg]')
axPhase.grid(True, which='major', linestyle='-')
axPhase.grid(True, which='minor', linestyle='--', alpha=0.5)


# %% ==========================================================================
# High-pass filter

R = 1e3
C = 1e-6

w = np.logspace(0, 6, 1000)

Highpass = 1/(1 - 1j/(w*R*C))

fig1 = plt.figure()

axMag = fig1.add_subplot(211)
axMag.loglog(w, np.abs(Highpass))
axMag.set_ylabel('Magnitude')
axMag.grid(True, which='major', linestyle='-')
axMag.grid(True, which='minor', linestyle='--', alpha=0.5)

axPhase = fig1.add_subplot(212, sharex=axMag)
axPhase.semilogx(w, np.angle(Highpass, deg=True))
axPhase.set_xlabel('Omega [rad/s]')
axPhase.set_ylabel('Phase [deg]')
axPhase.grid(True, which='major', linestyle='-')
axPhase.grid(True, which='minor', linestyle='--', alpha=0.5)


# %% ==========================================================================
















