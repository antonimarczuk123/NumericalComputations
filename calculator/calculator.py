# %% ==========================================================================
# Importing libraries

import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as plt


# %% ==========================================================================
# Phasor calculation

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
# Inverting amplifier

Vin, Vout, Vs, Z1, Z2, I, A = sp.symbols('Vin Vout Vs Z1 Z2 I A')

eq1 = sp.Eq(Vin - Vs, Z1*I)
eq2 = sp.Eq(Vs - Vout, Z2*I)
eq3 = sp.Eq(Vout, -A*Vs)

sol = sp.solve((eq1, eq2, eq3), (Vout, Vs, I))
Vout, Vs, I = sol[Vout], sol[Vs], sol[I]

print()
print(f"Vout = {Vout}")
print(f"Vs = {Vs}")
print(f"I = {I}")

# ideal A --> infinity

Vout_ideal = sp.simplify(sp.limit(Vout, A, sp.oo))
Vs_ideal = sp.simplify(sp.limit(Vs, A, sp.oo))
I_ideal = sp.simplify(sp.limit(I, A, sp.oo))

print()
print(f"Vout_ideal = {Vout_ideal}")
print(f"Vs_ideal = {Vs_ideal}")
print(f"I_ideal = {I_ideal}")


# %% ==========================================================================
# Non-inverting amplifier

Vin, Vout, Vs, Z1, Z2, I, A = sp.symbols('Vin Vout Vs Z1 Z2 I A')

eq1 = sp.Eq(Vs, Z1*I)
eq2 = sp.Eq(Vout - Vs, Z2*I)
eq3 = sp.Eq(Vout, A*(Vin - Vs))

sol = sp.solve((eq1, eq2, eq3), (Vout, Vs, I))
Vout, Vs, I = sol[Vout], sol[Vs], sol[I]

print()
print(f"Vout = {Vout}")
print(f"Vs = {Vs}")
print(f"I = {I}")

# ideal A --> infinity

Vout_ideal = sp.simplify(sp.limit(Vout, A, sp.oo))
Vs_ideal = sp.simplify(sp.limit(Vs, A, sp.oo))
I_ideal = sp.simplify(sp.limit(I, A, sp.oo))

print()
print(f"Vout_ideal = {Vout_ideal}")
print(f"Vs_ideal = {Vs_ideal}")
print(f"I_ideal = {I_ideal}")


# %% ==========================================================================


w = np.logspace(7.7, 7.9, 5000)
C0 = 5e-12
C1 = 0.01e-12
L1 =  25e-3
R1 = 50

Z0 = -1J/(w*C0)
Z1 = R1 + 1j*w*L1 - 1j/(w*C1)

Y = 1/Z0 + 1/Z1
Z = 1/Y

fig1 = plt.figure()

axMag = fig1.add_subplot(211)
axMag.loglog(w, Z.real)
axMag.set_ylabel('Magnitude')
axMag.minorticks_on()
axMag.grid(True, which='major', linestyle='-')
axMag.grid(True, which='minor', linestyle='--', alpha=0.5)

axPhase = fig1.add_subplot(212, sharex=axMag)
axPhase.semilogx(w, Z.imag)
axPhase.set_xlabel('Omega [rad/s]')
axPhase.set_ylabel('Phase [deg]')
axPhase.minorticks_on()
axPhase.grid(True, which='major', linestyle='-')
axPhase.grid(True, which='minor', linestyle='--', alpha=0.5)









