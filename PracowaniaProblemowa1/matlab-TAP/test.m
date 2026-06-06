clear; clc; close all;

% 1. PARAMETRY I STAŁE
ro = 1e6; roc = 1e6; cp = 1; cpc = 1;
k0 = 1e10; E_R = 8330.1; h = 130e6; a = 0.516e6; b = 0.5;
V = 1; Fin = 1; F = 1; Ts = 0.01;

% Zawężona, realistyczna przestrzeń dla wyznaczenia ekstremów wag
x2_grid = 390:2:420;   % Temperatura wokół punktu pracy
u2_grid = 10:2:30;     % Przepływ chłodzący
z2_grid = 300:2:320;   % Temperatura zasilania płaszcza

beta1_fun = @(x2) exp(-E_R / x2);
beta2_fun = @(x2,u2,z2) a * u2^b * (z2 - x2) / (u2 + (a * u2^b)/(2 * roc * cpc));

% Wyznaczenie granic
b1_all = arrayfun(beta1_fun, x2_grid);
beta1_min = min(b1_all); beta1_max = max(b1_all); beta1_avg = (beta1_max + beta1_min) / 2;

beta2_min = Inf; beta2_max = -Inf;
for x_g = x2_grid; for u_g = u2_grid; for z_g = z2_grid
    val = beta2_fun(x_g, u_g, z_g);
    if val < beta2_min, beta2_min = val; end
    if val > beta2_max, beta2_max = val; end
end; end; end
beta2_avg = (beta2_max + beta2_min) / 2;

% 2. FUNKCJE PRZYNALEŻNOŚCI (Zoptymalizowane)
mf_b1 = @(b) [ (b <= beta1_min) + (beta1_avg - b)/(beta1_avg - beta1_min)*(b > beta1_min && b < beta1_avg);
               (b - beta1_min)/(beta1_avg - beta1_min)*(b >= beta1_min && b < beta1_avg) + (beta1_max - b)/(beta1_max - beta1_avg)*(b >= beta1_avg && b < beta1_max);
               (b - beta1_avg)/(beta1_max - beta1_avg)*(b >= beta1_avg && b < beta1_max) + (b >= beta1_max) ];

mf_b2 = @(b) [ (b <= beta2_min) + (beta2_avg - b)/(beta2_avg - beta2_min)*(b > beta2_min && b < beta2_avg);
               (b - beta2_min)/(beta2_avg - beta2_min)*(b >= beta2_min && b < beta2_avg) + (beta2_max - b)/(beta2_max - beta2_avg)*(b >= beta2_avg && b < beta2_max);
               (b - beta2_avg)/(beta2_max - beta2_avg)*(b >= beta2_avg && b < beta2_max) + (b >= beta2_max) ];

% 3. GENEROWANIE MACIERZY LOKALNYCH (Rozszerzony Stan)
A = cell(3,3); B = cell(3,3); E = [0, 0; Fin/V, 0];
A_b1 = {beta1_min, beta1_avg, beta1_max};
B_b2 = {beta2_min, beta2_avg, beta2_max};

nx = 2; nu = 2; nz = 2;
N = 15; Nu = 3; % Skrócone horyzonty dla stabilności numerycznej

% Komórki na macierze syntezy MPC
K_cell = cell(3,3);

for r = 1:3
    for c = 1:3
        b1_v = A_b1{r}; b2_v = B_b2{c};
        A_loc = [-F/V - k0 * b1_v, 0; (h*k0)/(ro*cp) * b1_v, -F/V];
        B_loc = [Fin/V, 0; 0, b2_v];
        
        % Dyskretyzacja (Tustin)
        tmp = eye(nx) - Ts/2 * A_loc;
        Ad = tmp \ (eye(nx) + Ts/2 * A_loc);
        Bd = tmp \ (Ts * B_loc);
        
        % Konstrukcja modelu rozszerzonego: xi_k+1 = F_ed*xi_k + G_ed*du_k
        F_ed = [Ad, Bd; zeros(nu, nx), eye(nu)];
        G_ed = [Bd; eye(nu)];
        C_ed = [eye(nx), zeros(nx, nu)];
        
        % Budowa macierzy predykcji rozszerzonej (M i P)
        M = zeros(N*nx, Nu*nu);
        P = zeros(N*nx, nx+nu);
        
        for i = 1:N
            P((i-1)*nx+1:i*nx, :) = C_ed * (F_ed^i);
            for j = 1:Nu
                if i >= j
                    M((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = C_ed * (F_ed^(i-j)) * G_ed;
                end
            end
        end
        
        % Wagi MPC
        Q = kron(eye(N), diag([5000, 5]));      % Waga na błąd stanu (stężenie silniej)
        R = kron(eye(Nu), diag([10, 0.1]));     % Waga na przyrosty sterowania
        
        % Obliczenie wzmocnienia regulatora
        K_mpc = (M' * Q * M + R) \ (M' * Q);
        K_cell{r,c} = K_mpc(1:nu, :); % Pobieramy tylko pierwszy krok sterowania
        
        % Zapisujemy P do struktury powiązanej
        P_cell{r,c} = P;
    end
end

% 4. PARAMETRY SYMULACJI
Tend = 15; N_sim = floor(Tend/Ts); t = linspace(0, Tend, N_sim);
u_min = [0.01; 0.1]; u_max = [10; 80];
du_min = [-0.5; -5]; du_max = [0.5; 5];

x = zeros(nx, N_sim); u = zeros(nu, N_sim);
x(:,1) = [0.16; 405]; u(:,1) = [2; 15];
z0 = [343; 310];

% Trajektorie referencyjne i zakłócenia
x_ref = [ x(:,1) * ones(1, N_sim) + 0.04 * (t >= 1) - 0.04 * (t >= 7);
          x(2,1) * ones(1, N_sim) - 10 * (t >= 4) + 5 * (t >= 10) ];
z = [ z0(1) * ones(1, N_sim) + 5 * (t >= 3) - 10 * (t >= 12);
      z0(2) * ones(1, N_sim) + 5 * (t >= 8) ];

% Równanie nieliniowe obiektu (CSTR)
f_nonlin = @(x, u, z) [
    (1/V) * (Fin*u(1) - F*x(1) - V * k0 * exp(-E_R/x(2)) * x(1));
    (1/(V*ro*cp)) * (Fin*ro*cp*z(1) - F*ro*cp*x(2) + V*h*k0*exp(-E_R/x(2))*x(1) ...
        - (a * u(2)^(b+1) / (u(2) + (a * u(2)^b) / (2 * roc * cpc))) * (x(2) - z(2)))
];

% 5. PĘTLA SYMULACJI
n_sim = 10; h_step = Ts/n_sim;

for k = 1:N_sim-1
    x_curr = x(:, k);
    u_prev = u(:, k);
    z_curr = z(:, k);
    
    % Aktualne wartości beta
    b1 = beta1_fun(x_curr(2));
    b2 = beta2_fun(x_curr(2), u_prev(2), z_curr(2));
    
    % Wyznaczenie wag rozmytych
    w1 = mf_b1(b1); w2 = mf_b2(b2);
    W = w1 * w2'; % Macierz wag 3x3
    sum_W = sum(W(:));
    if sum_W == 0, sum_W = 1; W(2,2) = 1; end % Zabezpieczenie numeryczne
    
    % Przyszła referencja na horyzoncie (Poprawiony Błąd 1!)
    Xref = zeros(N*nx, 1);
    for idx = 1:N
        Xref((idx-1)*nx+1 : idx*nx) = x_ref(:, min(k + idx, N_sim));
    end
    
    % Rozszerzony wektor stanu dla regulatora
    xi = [x_curr; u_prev];
    
    % Agregacja prawa sterowania TS
    du = [0; 0];
    for r = 1:3
        for c = 1:3
            weight = W(r,c) / sum_W;
            if weight > 0.001
                % Wyznaczenie swobodnej odpowiedzi dla danego submodelu
                X0 = P_cell{r,c} * xi;
                % Przyrost sterowania
                du_loc = K_cell{r,c} * (Xref - X0);
                du = du + weight * du_loc;
            end
        end
    end
    
    % Ograniczenia przyrostu i wartości sterowania
    du = max(du_min, min(du_max, du));
    u_curr = u_prev + du;
    u_curr = max(u_min, min(u_max, u_curr));
    u(:, k+1) = u_curr;
    
    % Integracja numeryczna RK4 (Obiekt nieliniowy)
    x_next = x_curr;
    for ii = 1:n_sim
        k1 = f_nonlin(x_next, u_curr, z_curr);
        k2 = f_nonlin(x_next + 0.5*h_step*k1, u_curr, z_curr);
        k3 = f_nonlin(x_next + 0.5*h_step*k2, u_curr, z_curr);
        k4 = f_nonlin(x_next + h_step*k3, u_curr, z_curr);
        x_next = x_next + (h_step/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    x(:, k+1) = x_next;
end

% 6. WIZUALIZACJA
figure('Name', 'Wyniki MPCS Takagi-Sugeno', 'Position', [100 100 1000 700]);
subplot(2,2,1); plot(t, x(1,:), 'r', t, x_ref(1,:), 'k--', 'LineWidth', 1.5);
title('Stężenie C_A'); xlabel('Czas (min)'); ylabel('kmol/m^3'); grid on;
subplot(2,2,2); plot(t, x(2,:), 'b', t, x_ref(2,:), 'k--', 'LineWidth', 1.5);
title('Temperatura reaktora T'); xlabel('Czas (min)'); ylabel('K'); grid on;
subplot(2,2,3); stairs(t, u(1,:), 'g', 'LineWidth', 1.5);
title('Sterowanie C_{Ain}'); xlabel('Czas (min)'); grid on;
subplot(2,2,4); stairs(t, u(2,:), 'm', 'LineWidth', 1.5);
title('Sterowanie F_C (Przepływ)'); xlabel('Czas (min)'); grid on;