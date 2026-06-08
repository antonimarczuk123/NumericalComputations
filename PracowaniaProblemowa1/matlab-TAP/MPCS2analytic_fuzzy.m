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

Tend = 50; % simulation time [min]
N_sim = floor(Tend/Ts); % number of time steps

n_sim = 100; % number of RK4 steps per Ts to be more accurate
h_step = Ts/n_sim; % integration step size

t = linspace(0, Tend, N_sim); % time vector

x = [0.16; 405] * ones(1, N_sim);
u = [2; 15] * ones(1, N_sim);

x_ref = [
    % CA reference trajectory
    0.16 * ones(1, N_sim) - 0.1 * (t >= 1) + 0.05 * (t >= 10) + 0.05 * (t >= 35);
    % T reference trajectory
    405 * ones(1, N_sim) + 25 * (t >= 15) - 10 * (t >= 30) - 15 * (t >= 40);
];

z = [
    % Tin disturbance trajectory
    340 * ones(1, N_sim) - 10 * (t >= 5) + 20 * (t >= 30);
    % TCin disturbance trajectory
    310 * ones(1, N_sim) + 10 * (t >= 20) - 20 * (t >= 40);
];


% ================================================================================
% MPCS simulation loop

du_val = cell(3, 3);
mf_val = cell(3, 3);
X0 = zeros(N*nx, 1);

for k = 2:N_sim-1
    x_ref_curr = x_ref(:, k);

    x_prev = x(:, k-1); % we need to remember that
    x_curr = x(:, k);   % we need to measure that

    u_prev = u(:, k-1); % we need to remember that

    z_prev = z(:, k-1); % we need to remember that
    z_curr = z(:, k);   % we need to measure that

    % ---------- MPCS control law
    
    mf_val_sum = 0;
    for i = 1:3
        for j = 1:3
            mf_val{i,j} = mf{i,j}(x_curr(2), u_prev(2));
            mf_val_sum = mf_val_sum + mf_val{i,j};

            d = (x_curr - xp{i,j}) - (Ad{i,j} * (x_prev - xp{i,j}) + Bd{i,j} * (u_prev - up{i,j}) + Ed{i,j} * (z_prev - zp{i,j}) + alfad{i,j});

            X0 = At{i,j} * (x_curr - xp{i,j}) + Vt{i,j} * (Bd{i,j} * (u_prev - up{i,j}) + Ed{i,j} * (z_curr - zp{i,j}) + alfad{i,j} + d);

            Xref = repmat(x_ref_curr - xp{i,j}, N, 1);

            du_val{i,j} = K1{i,j} * (Xref - X0);
        end
    end

    du = [0;0];
    for i = 1:3
        for j = 1:3
            du = du + (mf_val{i,j}/mf_val_sum) * du_val{i,j};
        end
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
    x(:, k+1) = x_next + 0.0002 * [CA_p; T_p] * randn(); % add some noise to make it more realistic
    % x(:, k+1) = x_next;
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





