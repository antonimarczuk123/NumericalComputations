# %% =================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Parametry objektu
ro = 1e6; # g/m^3
roc = 1e6; # g/m^3
cp = 1; # cal/(g*K)
cpc = 1; # cal/(g*K)
k0 = 1e10; # 1/min
E_R = 8330.1; # 1/K
h = 130e6; # cal/kmol
a = 0.516e6; # cal/(K*m^3)
b = 0.5; # -

# Stałe procesu
V = 1; # m^3
Fin = 1; # m^3/min
F = 1; # m^3/min

# Czas próbkowania
Ts = 0.0001

y0 = np.array([0.5, 350])

def CAin(t):
    return 2.0 + 2.0 * (t>1)

def FC(t):
    return 15.0 + 10 * (t>3)

def Tin(t):
    return 343

def TCin(t):
    return 310

CA0 = 0.16
T0 = 404
y0 = np.array([CA0, T0])

# dy/dt = f_c(y, u, z)
def f_c(t, y):
    CA = y[0]
    T = y[1]

    dy0_dt = (1/V) * (Fin*CAin(t) - F*CA - V * k0 * np.exp(-E_R/T) * CA)
    dy1_dt = (1/(V*ro*cp)) * (Fin*ro*cp*Tin(t) - F*ro*cp*T + V*h*k0*np.exp(-E_R/T)*CA - (a * FC(t)**(b+1) / (FC(t) + (a * FC(t)**b) / (2 * roc * cpc))) * (T - TCin(t)))

    return np.array([dy0_dt, dy1_dt])

def f_d(t, y):
    # RK4
    k1 = f_c(t, y)
    k2 = f_c(t + Ts/2, y + Ts/2 * k1)
    k3 = f_c(t + Ts/2, y + Ts/2 * k2)
    k4 = f_c(t + Ts, y + Ts * k3)
    return y + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

# Symulacja metodą Radau
time_span = (0, 10)
time_eval = np.arange(time_span[0], time_span[1], Ts)
sol = scipy.integrate.solve_ivp(f_c, time_span, y0, t_eval=time_eval, method='Radau')

# Symulacja metodą RK4
CA_rk4 = np.zeros_like(time_eval)
T_rk4 = np.zeros_like(time_eval)
y = y0
for i, t in enumerate(time_eval):
    CA_rk4[i] = y[0]
    T_rk4[i] = y[1]
    y = f_d(t, y)

# Rysowanie wykresów
fig1 = plt.figure()

ax = fig1.add_subplot(211)
ax.plot(sol.t, sol.y[0], label='CA(t)')
ax.plot(time_eval, CA_rk4, label='CA(t) RK4', linestyle='--')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)

ax = fig1.add_subplot(212)
ax.plot(sol.t, sol.y[1], label='T(t)')
ax.plot(time_eval, T_rk4, label='T(t) RK4', linestyle='--')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--', alpha=0.5)



