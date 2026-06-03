%% objSim.m
% ================================================================================
% Definition of the nonlinear system, and linearization around the operating point.
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

stepsCain = [-0.1 -0.04 -0.01 0.01 0.04 0.15 0.2];
stepsFc = [-3 -2 -1 -0.5 -0.25 0.25 0.5 1 1.75];
stepsTin = [-8 -5 -2 2 5 10 20];
stepsTCin = [-5 -2 2 5 10 15];

figure;
hold on;

for i=1:length(stepsCain)

% Control inputs
CAin = @(t) CAin_p + stepsCain(i) * (t > 1); 
FC = @(t) FC_p +0 * (t > 1);

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
hold on;
plot(t_nl, CAin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie CAin (kmol/m^3)');
title('Sterowanie - CAin');
grid on; grid minor;

subplot(3,2,4);
hold on;
plot(t_nl, FC(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie FC (m^3/min)');
title('Sterowanie - FC');
grid on; grid minor;

subplot(3,2,5);
hold on;
plot(t_nl, Tin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie Tin (K)');
title('Zakłócenie - Tin');
grid on; grid minor;

subplot(3,2,6);
hold on;
plot(t_nl, TCin(t_nl), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie TCin (K)');
title('Zakłócenie - TCin');
grid on; grid minor;

end


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
stairs(time_steps, x_disc(1,:), 'LineWidth',2);
legend('Model zlinearyzowany (ciągły)', 'Model zlinearyzowany (dyskretny - RK4)');
title('Odpowiedź układu - stężenie CA');
grid on; grid minor;

subplot(2,1,2);
plot(t_lin, x_lin(:,2), 'b-', 'LineWidth', 2); hold on;
stairs(time_steps, x_disc(2,:), 'LineWidth',2);
legend('Model zlinearyzowany (ciągły)', 'Model zlinearyzowany (dyskretny - RK4)');
title('Odpowiedź układu - temperatura T');
grid on; grid minor;


% ================================================================================
% Transfer functions of the linearized model

G = tf(ss(A,B,eye(2),zeros(2,2))) % for control inputs
Gz = tf(ss(A,E,eye(2),zeros(2,2))) % for disturbance inputs

% ================================================================================
% Transfer functions of the discretized linear model

Gd = tf(ss(Ad,Bd,eye(2),zeros(2,2), Ts*60)) % for control inputs
Gdz = tf(ss(Ad,Ed,eye(2),zeros(2,2), Ts*60)) % for disturbance inputs
% Note: Ts (sampling time) is specified in minutes not seconds, so we multiply by 60 to convert to seconds.


t_sim = (0:0.01:Tend)';
t_disc = (0:Ts:Tend)';

sys_ss_c = ss(A, B, eye(2), zeros(2,2));
sys_ss_z_c = ss(A, E, eye(2), zeros(2,2));

sys_ss_d = ss(Ad, Bd, eye(2), zeros(2,2), Ts);
sys_ss_z_d = ss(Ad, Ed, eye(2), zeros(2,2), Ts);
Gd.Ts = Ts; 
Gdz.Ts = Ts;

input_names = {'CAin', 'FC', 'Tin', 'TCin'};
all_steps = {stepsCain, stepsFc, stepsTin, stepsTCin};

for k = 1:4
    figure('NumberTitle', 'off');
    current_steps = all_steps{k};
    
    for s = 1:length(current_steps)
        u_step = current_steps(s);
        u_c = zeros(length(t_sim), 2);    
        u_d = zeros(length(t_disc), 2);
        z_c = zeros(length(t_sim), 2);   
        z_d = zeros(length(t_disc), 2);

        if k <= 2
            u_c(t_sim >= 1, k) = u_step;
            u_d(t_disc >= 1, k) = u_step;
        else
            z_c(t_sim >= 1, k-2) = u_step;
            z_d(t_disc >= 1, k-2) = u_step;
        end

        y_ss_c = lsim(sys_ss_c, u_c, t_sim) + lsim(sys_ss_z_c, z_c, t_sim);
        y_tf_c = lsim(G, u_c, t_sim) + lsim(Gz, z_c, t_sim);
        y_ss_d = lsim(sys_ss_d, u_d, t_disc) + lsim(sys_ss_z_d, z_d, t_disc);
        y_tf_d = lsim(Gd, u_d, t_disc) + lsim(Gdz, z_d, t_disc);

        subplot(2,1,1); hold on;
        plot(t_sim, y_ss_c(:,1) + CA_p, 'b-', 'LineWidth', 1.5);
        plot(t_sim, y_tf_c(:,1) + CA_p, 'r--', 'LineWidth', 1);   
        stairs(t_disc, y_ss_d(:,1) + CA_p, 'g-', 'LineWidth', 1);  
        plot(t_disc, y_tf_d(:,1) + CA_p, 'm:', 'LineWidth', 1.5);  

        subplot(2,1,2); hold on;
        plot(t_sim, y_ss_c(:,2) + T_p, 'b-', 'LineWidth', 1.5);
        plot(t_sim, y_tf_c(:,2) + T_p, 'r--', 'LineWidth', 1);
        stairs(t_disc, y_ss_d(:,2) + T_p, 'g-', 'LineWidth', 1);
        plot(t_disc, y_tf_d(:,2) + T_p, 'm:', 'LineWidth', 1.5);
    end
    
    subplot(2,1,1); grid on; ylabel('CA [kmol/m^3]');
    title(['Odpowiedzi na skoki ', input_names{k}]);
    
    subplot(2,1,2); grid on; ylabel('T [K]'); xlabel('Czas [min]');
    legend('Model Ciągły', 'Transmitancja Ciągła', 'Model Dyskretny', 'Transmitancja Dyskretna', ...
               'Orientation', 'horizontal', 'Location', 'southoutside');
end