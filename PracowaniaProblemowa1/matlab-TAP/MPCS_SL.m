%% ================================================================================
clear; clc;

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

% Sampling time
Ts = 0.01;

CAin_min = 0.01; CAin_max = 10; % kmol/m^3
dCAin_min = -0.5; dCAin_max = 0.5; % kmol/(m^3*min)

FC_min = 0.1; FC_max = 80; % m^3/min
dFC_min = -5; dFC_max = 5; % m^3/(min^2)

u_min = [CAin_min; FC_min];
u_max = [CAin_max; FC_max];

du_min = [dCAin_min; dFC_min];
du_max = [dCAin_max; dFC_max];

% initial state, control, disturbance
CA0 = 0.16; % kmol/m^3
T0 = 405; % K
Tin0 = 343; % K
TCin0 = 310; % K
CAin0 = 2; % kmol/m^3
FC0 = 15; % m^3/min

x0 = [CA0; T0];
u0 = [CAin0; FC0];
z0 = [Tin0; TCin0];

% object equations (continuous and nonlinear)
ff = @(CA, T, CAin, FC, Tin, TCin) [
    (1/V) * (Fin*CAin - F*CA - V * k0 * exp(-E_R/T) * CA);
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin - F*ro*cp*T + V*h*k0*exp(-E_R/T)*CA - (a * FC^(b+1) / (FC + (a * FC^b) / (2 * roc * cpc))) * (T - TCin))
];

f = @(x, u, z) ff(x(1), x(2), u(1), u(2), z(1), z(2));

Tend = 50; % simulation time [min]
N_sim = floor(Tend/Ts); % number of time steps

n_sim = 100; % number of RK4 steps per Ts to be more accurate
h_step = Ts/n_sim; % integration step size

t = linspace(0, Tend, N_sim); % time vector

x = x0 * ones(1, N_sim);
u = u0 * ones(1, N_sim);

x_ref = [
    % CA reference trajectory
    CA0 * ones(1, N_sim) + 0.1 * (t >= 1) - 0.05 * (t >= 10) - 0.05 * (t >= 35);
    % T reference trajectory
    T0 * ones(1, N_sim) - 10 * (t >= 15) - 10 * (t >= 30) + 30 * (t >= 40);
];

z = [
    % Tin disturbance trajectory
    Tin0 * ones(1, N_sim) + 30 * (t >= 5) - 60 * (t >= 30);
    % TCin disturbance trajectory
    TCin0 * ones(1, N_sim) + 30 * (t >= 20) - 60 * (t >= 40);
];


% ================================================================================
% MPCS simulation loop

A = zeros(2, 2);
B = zeros(2, 2);
E = zeros(2, 2);

Ad = zeros(2, 2);
Bd = zeros(2, 2);
Ed = zeros(2, 2);

N = 50; % prediction horizon
Nu = 5; % control horizon
nx = 2; % number of states
nu = 2; % number of controls

lambda = 20; % control penalty
Q = kron(eye(N), diag([10000, 1]));
R = lambda * kron(eye(Nu), diag([20, 0.1]));

tmp1 = zeros(nx, nx);
tmp2 = zeros(N*nx, nx);
tmp3 = eye(nx);
M = zeros(N*nx, Nu*nu);
[nn, ~] = size(M);
K = zeros(Nu*nu, N*nx);
K1 = zeros(nu, N*nx);
X0 = zeros(N*nx, 1);

for k = 2:N_sim-1
    x_ref_curr = x_ref(:, k);

    x_prev = x(:, k-1); % we need to remember that
    x_curr = x(:, k);   % we need to measure that

    u_prev = u(:, k-1); % we need to remember that

    z_prev = z(:, k-1); % we need to remember that
    z_curr = z(:, k);   % we need to measure that

    % linearization around current point

    CA_p = x_curr(1); T_p = x_curr(2);
    CAin_p = u_prev(1); FC_p = u_prev(2);
    Tin_p = z_curr(1); TCin_p = z_curr(2);

    xp = [CA_p; T_p];
    up = [CAin_p; FC_p];
    zp = [Tin_p; TCin_p];

    Df_CA_p     =   (ff(CA_p + 1e-8, T_p, CAin_p, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_T_p      =   (ff(CA_p, T_p + 1e-8, CAin_p, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_CAin_p   =   (ff(CA_p, T_p, CAin_p + 1e-8, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_FC_p     =   (ff(CA_p, T_p, CAin_p, FC_p + 1e-8, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_Tin_p    =   (ff(CA_p, T_p, CAin_p, FC_p, Tin_p + 1e-8, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_TCin_p   =   (ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p + 1e-8) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;

    A = [Df_CA_p, Df_T_p];
    B = [Df_CAin_p, Df_FC_p];
    E = [Df_Tin_p, Df_TCin_p];

    alfa = ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p);

    % discretization using trapezoidal rule

    tmp1 = eye(2) - Ts/2 * A;
    Ad = tmp1 \ (eye(2) + Ts/2 * A);
    Bd = tmp1 \ (Ts * B);
    Ed = tmp1 \ (Ts * E);
    alfad = tmp1 \ (alfa * Ts);

    % MPCS

    tmp2 = zeros(N*nx, nx);
    tmp3 = eye(nx);
    for i=1:N
        tmp2((i-1)*nx+1:i*nx, :) = tmp3;
        tmp3 = Ad * tmp3 + eye(nx);
    end

    M = zeros(N*nx, Nu*nu);
    for i=1:Nu
        M((i-1)*nx+1 : end, (i-1)*nu+1 : i*nu) = tmp2(1 : nn - (i-1)*nx, :) * Bd;
    end

    K = (M' * Q * M + R) \ (M' * Q);
    K1 = K(1:nu, :);

    d = (x_curr - xp) - (Ad * (x_prev - xp) + Bd * (u_prev - up) + Ed * (z_prev - zp) + alfad);

    xtmp = x_curr - xp; % liczymy trajektorię swobodną
    for i=1:N
        xtmp = Ad * xtmp + Bd * (u_prev - up) + Ed * (z_curr - zp) + alfad + d;
        X0((i-1)*nx+1 : i*nx) = xtmp;
    end

    Xref = repmat(x_ref_curr - xp, N, 1);

    % compute control action

    du = K1 * (Xref - X0);
    du = max(du_min, min(du_max, du));
    u_curr = u_prev + du;
    u_curr = max(u_min, min(u_max, u_curr));

    u(:, k) = u_curr;

    % ---------- Simulation step
    x_next = x_curr;
    for i = 1:n_sim
        k1 = f(x_next, u_curr, z_curr);
        k2 = f(x_next + 0.5*h_step*k1, u_curr, z_curr);
        k3 = f(x_next + 0.5*h_step*k2, u_curr, z_curr);
        k4 = f(x_next + h_step*k3, u_curr, z_curr);
        x_next = x_next + (h_step/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    % x(:, k+1) = x_next + 0.0002 * [CA_p; T_p] * randn(); % add some noise to make it more realistic
    x(:, k+1) = x_next;
end

% ================================================================================
% plot results

figure;

subplot(3,2,1);
plot(t(1:end-1), x(1,1:end-1), 'r-', 'LineWidth', 2); hold on;
plot(t(1:end-1), x_ref(1,1:end-1), 'k--', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Stężenie CA (kmol/m^3)');
title('Odpowiedź układu - stężenie CA');
grid on; grid minor;

subplot(3,2,2);
plot(t(1:end-1), x(2,1:end-1), 'r-', 'LineWidth', 2); hold on;
plot(t(1:end-1), x_ref(2,1:end-1), 'k--', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Temperatura T (K)');
title('Odpowiedź układu - temperatura T');
grid on; grid minor;

subplot(3,2,3);
stairs(t(1:end-1), u(1,1:end-1), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie CAin (kmol/m^3)');
title('Sterowanie - CAin');
grid on; grid minor;

subplot(3,2,4);
stairs(t(1:end-1), u(2,1:end-1), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Sterowanie FC (m^3/min)');
title('Sterowanie - FC');
grid on; grid minor;

subplot(3,2,5);
plot(t(1:end-1), z(1,1:end-1), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie Tin (K)');
title('Zakłócenie - Tin');
grid on; grid minor;

subplot(3,2,6);
plot(t(1:end-1), z(2,1:end-1), 'k-', 'LineWidth', 2);
xlabel('Czas (min)'); ylabel('Zakłócenie TCin (K)');
title('Zakłócenie - TCin');
grid on; grid minor;





