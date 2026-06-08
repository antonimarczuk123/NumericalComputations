%% ================================================================================
% Skrypt do badania nieliniowego objektu i jakości linearyzacji w punkcie równowagi.

clear; clc;
format short e;

% Parameters
ro = 1e6; % g/m^3
roc = 1e6; % g/m^3
cp = 1; % cal/(g*K)
cpc = 1; % cal/(g*K)
k0 = 1e10; % 1/min
E_R = 8330.1; % 1/K
h = 130e6; % cal/kmol
a = 0.516e6; % cal/(K*m^3)
b = 0.5; % -

% Constants
% Fin = F, so V is constant
V = 1; % m^3
Fin = 1; % m^3/min
F = 1; % m^3/min

% Controlled variables: CA, T
% Manipulated variables: CAin, FC
% Disturbance variables: Tin, TCin

% State equations
% dCA/dt = f1(CA, T, CAin, FC, Tin, TCin)
% dT/dt = f2(CA, T, CAin, FC, Tin, TCin)
f = @(CA, T, CAin, FC, Tin, TCin) [
    (1/V) * (Fin*CAin - F*CA - V * k0 * exp(-E_R/T) * CA);
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin - F*ro*cp*T + V*h*k0*exp(-E_R/T)*CA - (a * FC^(b+1) / (FC + (a * FC^b) / (2 * roc * cpc))) * (T - TCin))
];

% Operating point
CA_p = 0.1597172015358418; % kmol/m^3  - computed in findOpPoint.m
T_p = 404.7356680993987; % K          - computed in findOpPoint.m
CAin_p = 2; % kmol/m^3
FC_p = 15; % m^3/min
Tin_p = 343; % K
TCin_p = 310; % K

% Check if the operating point is equilibrium point (f should be zero)
f_p = f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p);
fprintf("W punkcie równowagi pochodna powinna wynosić zero:\n");
disp(f_p);

% Linearization
% [dCA/dt; dT/dt] = A*[CA - CA_p; T - T_p] + B*[CAin - CAin_p; FC - FC_p] + E*[Tin - Tin_p; TCin - TCin_p]
% where A, B, E are Jacobian matrices of f with respect to state, control, and disturbance variables, evaluated at the operating point.

Df_CA_p     =   (f(CA_p + 1e-8, T_p, CAin_p, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
Df_T_p      =   (f(CA_p, T_p + 1e-8, CAin_p, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
Df_CAin_p   =   (f(CA_p, T_p, CAin_p + 1e-8, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
Df_FC_p     =   (f(CA_p, T_p, CAin_p, FC_p + 1e-8, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
Df_Tin_p    =   (f(CA_p, T_p, CAin_p, FC_p, Tin_p + 1e-8, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
Df_TCin_p   =   (f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p + 1e-8) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;

A = [Df_CA_p, Df_T_p];
B = [Df_CAin_p, Df_FC_p];
E = [Df_Tin_p, Df_TCin_p];

fprintf("Macierz A (pochodne względem stanu):\n");
disp(A);

fprintf("Macierz B (pochodne względem wielkości sterujących):\n");
disp(B);

fprintf("Macierz E (pochodne względem wielkości zakłócających):\n");
disp(E);

% ================================================================================
% Simulation of the nonlinear and linearized model using ode45 and comparison of the results.

Tend = 20; % min

% Control inputs
CAin = @(t) CAin_p + 0 * (t > 1); 
FC = @(t) FC_p + 30 * (t > 1);

% Disturbances
Tin = @(t) Tin_p + 0 * (t > 3);
TCin = @(t) TCin_p + 0 * (t > 5);

% Nonlinear function for ode45, x = [CA; T]
f_nonlin_ode = @(t, x) [
    (1/V) * (Fin*CAin(t) - F*x(1) - V * k0 * exp(-E_R/x(2)) * x(1));
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin(t) - F*ro*cp*x(2) + V*h*k0*exp(-E_R/x(2))*x(1) - (a * FC(t)^(b+1) / (FC(t) + (a * FC(t)^b) / (2 * roc * cpc))) * (x(2) - TCin(t)))
];

% Linearized model for ode45, x = [CA; T]
f_lin_ode = @(t, x) A*(x - [CA_p; T_p]) + B*[CAin(t) - CAin_p; FC(t) - FC_p] + E*[Tin(t) - Tin_p; TCin(t) - TCin_p];

% Simulation with ode45
[t_nl, x_nl] = ode45(f_nonlin_ode, [0 Tend], [CA_p; T_p], odeset('RelTol',1e-8,'AbsTol',1e-10));
[t_lin, x_lin] = ode45(f_lin_ode, [0 Tend], [CA_p; T_p], odeset('RelTol',1e-8,'AbsTol',1e-10));

figure;

subplot(3,2,1);
plot(t_nl, x_nl(:,1), 'r-', 'LineWidth', 2); hold on;
plot(t_lin, x_lin(:,1), 'b--', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Stężenie CA (kmol/m^3)');
legend('Model nieliniowy', 'Model zlinearyzowany');
title('Odpowiedź układu - stężenie CA');
grid on; grid minor;

subplot(3,2,2);
plot(t_nl, x_nl(:,2), 'r-', 'LineWidth', 2); hold on;
plot(t_lin, x_lin(:,2), 'b--', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Temperatura T (K)');
legend('Model nieliniowy', 'Model zlinearyzowany');
title('Odpowiedź układu - temperatura T');
grid on; grid minor;

subplot(3,2,3);
plot(t_nl, CAin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie CAin (kmol/m^3)');
title('Sterowanie - CAin');
grid on; grid minor;

subplot(3,2,4);
plot(t_nl, FC(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie FC (m^3/min)');
title('Sterowanie - FC');
grid on; grid minor;

subplot(3,2,5);
plot(t_nl, Tin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie Tin (K)');
title('Zakłócenie - Tin');
grid on; grid minor;

subplot(3,2,6);
plot(t_nl, TCin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie TCin (K)');
title('Zakłócenie - TCin');
grid on; grid minor;


% ================================================================================
% Find discretization of linearized model with RK4

Ts = 0.05; % sampling time [min]

% x - state (and output) <- x = [CA; T]
% u - control input <- u = [CAin; FC]
% z - disturbance <- z = [Tin; TCin]

% dx/dt = A*(x-xp) + B*(u-up) + E*(z-zp)
% x(k+1) = Ad*(x(k)-xp) + Bd*(u(k)-up) + Ed*(z(k)-zp) + xp

% where Ad, Bd, Ed:
% Ad = eye(size(A)) + A*Ts + (A^2)*(Ts^2)/2 + (A^3)*(Ts^3)/6 + (A^4)*(Ts^4)/24;
% Bd = B*Ts + A*B*(Ts^2)/2 + A^2*B*(Ts^3)/6 + A^3*B*(Ts^4)/24;
% Ed = E*Ts + A*E*(Ts^2)/2 + A^2*E*(Ts^3)/6 + A^3*E*(Ts^4)/24;

% Fast way to compute the above:
A2 = A*A;
A3 = A2*A;
Tmp = Ts*eye(2) + A*(Ts^2)/2 + A2*(Ts^3)/6 + A3*(Ts^4)/24;
Ad = eye(2) + A*Tmp;
Bd = Tmp*B;
Ed = Tmp*E;

fprintf("Macierz Ad (RK4):\n");
disp(Ad);

fprintf("Macierz Bd (RK4):\n");
disp(Bd);

fprintf("Macierz Ed (RK4):\n");
disp(Ed);

discrete_step = @(x,k) Ad*(x - [CA_p; T_p]) + Bd*[CAin(k*Ts) - CAin_p; FC(k*Ts) - FC_p] + Ed*[Tin(k*Ts) - Tin_p; TCin(k*Ts) - TCin_p] + [CA_p; T_p];

time_steps = 0:Ts:Tend;
x_disc = zeros(2, length(time_steps));
x_disc(:,1) = [CA_p; T_p]; % initial condition (operating point)
for k = 1:length(time_steps)-1
    x_disc(:,k+1) = discrete_step(x_disc(:,k), k);
end

figure;

subplot(2,1,1);
plot(t_lin, x_lin(:,1), 'b-', 'LineWidth', 2); hold on;
plot(time_steps, x_disc(1,:), 'r.', 'MarkerSize', 12);
legend('Model zlinearyzowany (ciągły)', 'Model zlinearyzowany (dyskretny - RK4)');
title('Odpowiedź układu - stężenie CA');
grid on; grid minor;

subplot(2,1,2);
plot(t_lin, x_lin(:,2), 'b-', 'LineWidth', 2); hold on;
plot(time_steps, x_disc(2,:), 'r.', 'MarkerSize', 12);
legend('Model zlinearyzowany (ciągły)', 'Model zlinearyzowany (dyskretny - RK4)');
title('Odpowiedź układu - temperatura T');
grid on; grid minor;




