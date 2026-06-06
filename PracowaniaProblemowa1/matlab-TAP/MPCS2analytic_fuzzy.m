%% ================================================================================
% Before running this file, run MPCS1.m to compute the MPCS matrices.
% Then run MPCS2analytic.m or MPCS2numeric to simulate the closed-loop system.

CAin_min = 0.01; CAin_max = 10; % kmol/m^3
dCAin_min = -0.5; dCAin_max = 0.5; % kmol/(m^3*min)

FC_min = 0.1; FC_max = 80; % m^3/min
dFC_min = -5; dFC_max = 5; % m^3/(min^2)

u_min = [CAin_min; FC_min];
u_max = [CAin_max; FC_max];

du_min = [dCAin_min; dFC_min];
du_max = [dCAin_max; dFC_max];

% initial state, control, disturbance
x0 = [0.16; 405];
u0 = [2; 15];
z0 = [343; 310];

% object equations (continuous and nonlinear)
f = @(x, u, z) [
    (1/V) * (Fin*u(1) - F*x(1) - V * k0 * exp(-E_R/x(2)) * x(1));
    (1/(V*ro*cp)) * (Fin*ro*cp*z(1) - F*ro*cp*x(2) + V*h*k0*exp(-E_R/x(2))*x(1) ...
        - (a * u(2)^(b+1) / (u(2) + (a * u(2)^b) / (2 * roc * cpc))) * (x(2) - z(2)))
];

Tend = 50; % simulation time [min]
N_sim = floor(Tend/Ts); % number of time steps

n_sim = 100; % number of RK4 steps per Ts to be more accurate
h_step = Ts/n_sim; % integration step size

t = linspace(0, Tend, N_sim); % time vector

x = x0 * ones(1, N_sim);
u = u0 * ones(1, N_sim);


x_ref = [
    % CA reference trajectory
    x0(1) * ones(1, N_sim) + 0.05 * (t >= 1) + 0 * (t >= 10) - 0.05 * (t >= 35);
    % T reference trajectory
    x0(2) * ones(1, N_sim) - 10 * (t >= 15) + 0 * (t >= 30);
];

z = [
    % Tin disturbance trajectory
    z0(1) * ones(1, N_sim) + 0 * (t >= 5) - 0 * (t >= 30);
    % TCin disturbance trajectory
    z0(2) * ones(1, N_sim) + 0 * (t >= 20) + 0 * (t >= 40);
];


% ================================================================================
% MPCS simulation loop

A = zeros(2, 2);
B = zeros(2, 2);
E = [0, 0; Fin/V, 0];

Ad = zeros(2, 2);
Bd = zeros(2, 2);
Ed = zeros(2, 2);

for k = 2:N_sim-1
    x_ref_curr = x_ref(:, k);

    x_prev = x(:, k-1); % we need to remember that
    x_curr = x(:, k);   % we need to measure that

    u_prev = u(:, k-1); % we need to remember that

    z_prev = z(:, k-1); % we need to remember that
    z_curr = z(:, k);   % we need to measure that

    % ---------- MPCS control law

    b1 = beta1_fun(x_curr(2));
    b2 = beta2_fun(x_curr(2), u_prev(2), z_curr(2));

    A = [-F/V - k0 * b1, 0; (h*k0)/(ro*cp) * b1, -F/V];
    B = [Fin/V, 0; 0, b2];





    mf_val_sum = 0;
    for ii = 1:9
        vk = x_curr - (Ad{ii}*x_prev + Bd{ii}*u_prev + Ed{ii}*z_prev);
        X0 = At{ii}*x_curr + Vt{ii} * (Bd{ii}*u_prev + Ed{ii}*z_curr + vk);
        Xref = repmat(x_ref_curr, N, 1);
        du_ii = K1{ii} * (Xref - X0);
        du_ii = max(du_min, min(du_max, du_ii));
        du_val{ii} = du_ii;

        mf_val{ii} = mf{ii}(b1, b2);
        mf_val_sum = mf_val_sum + mf_val{ii};
    end

    du = [0;0];
    for ii = 1:9
        du = du + (mf_val{ii}/mf_val_sum) * du_val{ii};
    end

    du = max(du_min, min(du_max, du));
    u_curr = u_prev + du;
    u_curr = max(u_min, min(u_max, u_curr));

    u(:, k) = u_curr;

    % ---------- Simulation step
    x_next = x_curr;
    for ii = 1:n_sim
        k1 = f(x_next, u_curr, z_curr);
        k2 = f(x_next + 0.5*h_step*k1, u_curr, z_curr);
        k3 = f(x_next + 0.5*h_step*k2, u_curr, z_curr);
        k4 = f(x_next + h_step*k3, u_curr, z_curr);
        x_next = x_next + (h_step/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
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





