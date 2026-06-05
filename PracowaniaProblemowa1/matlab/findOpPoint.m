%% findOpPoint.m
% ================================================================================
% Finding the operating point for the nonlinear system.
clear; clc;
format long e;

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

% Operating point - control and disturbances
CAin_p = 2.0; % kmol/m^3
FC_p = 7; % m^3/min
Tin_p = 343; % K
TCin_p = 310; % K

% CAin: 
% ---------------------------
% 2.0

% FC:
% ---------------------------
% 7
% 9
% 12
% 15
% 17
% 18

% CA:
% ---------------------------
% 3.487884655658222e-02 = 0.0348784655658222
% 5.557088024052338e-02 = 0.05557088024052338
% 9.821109463204293e-02 = 0.09821109463204293
% 1.597172021766315e-01 = 0.1597172021766315
% 2.169364570631375e-01 = 0.2169364570631375
% 1.828622864494349e+00 = 1.828622864494349

% T:
% ---------------------------
% 4.385550598263770e+02 = 438.5550598263770
% 4.278255516809578e+02 = 427.8255516809578
% 4.152091350393931e+02 = 415.2091350393931
% 4.047356680017397e+02 = 404.7356680017397
% 3.982004016808073e+02 = 398.2004016808073
% 3.280431883372068e+02 = 328.0431883372068


% Approximate values of the output at the operating point
CA_p = 0.16; % kmol/m^3
T_p = 405; % K

Tend = 1000; % min

% Nonlinear model for ode45, x = [CA; T]
f_ode = @(t, x) [
    (1/V) * (Fin*CAin_p - F*x(1) - V * k0 * exp(-E_R/x(2)) * x(1));
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin_p - F*ro*cp*x(2) + V*h*k0*exp(-E_R/x(2))*x(1) - (a * FC_p^(b+1) / (FC_p + (a * FC_p^b) / (2 * roc * cpc))) * (x(2) - TCin_p))
];

% symulacja ode45
% [t, x] = ode45(f_ode, [0 Tend], [CA_p; T_p], odeset('RelTol',1e-8,'AbsTol',1e-10));
[t, x] = ode15s(f_ode, [0 Tend], [CA_p; T_p], odeset('RelTol',1e-8,'AbsTol',1e-10));

fprintf("Punkt pracy (CA_p, T_p):\n");
disp(x(end,:));

figure;

subplot(2,1,1);
plot(t, x(:,1), 'r-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Stężenie CA (kmol/m^3)');
title('Odpowiedź układu - stężenie CA');
grid on; grid minor;

subplot(2,1,2);
plot(t, x(:,2), 'r-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Temperatura T (K)');
title('Odpowiedź układu - temperatura T');
grid on; grid minor;










