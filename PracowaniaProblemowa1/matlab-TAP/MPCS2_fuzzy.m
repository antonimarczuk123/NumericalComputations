%% ================================================================================
% Skrypt przygotowujący zmienne i macierze dla rozmytego regulatora MPCS.
% Rozmycie mamy po T i FC, więc mamy 9 reguł rozmytych, a co za tym idzie 9 regulatorów MPCS.
% Każdy regulator MPCS jest zbudowany na podstawie liniowej aproksymacji obiektu:
% dx/dt = f(x, u, z) = f(xp, up, zp) + A*(x-xp) + B*(u-up) + E*(z-zp)
% Linearyzacja nie jest w punktach równowagi!

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

% object equations (continuous and nonlinear)
ff = @(CA, T, CAin, FC, Tin, TCin) [
    (1/V) * (Fin*CAin - F*CA - V * k0 * exp(-E_R/T) * CA);
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin - F*ro*cp*T + V*h*k0*exp(-E_R/T)*CA - (a * FC^(b+1) / (FC + (a * FC^b) / (2 * roc * cpc))) * (T - TCin))
];

f = @(x, u, z) ff(x(1), x(2), u(1), u(2), z(1), z(2));

CA_p = 0.2;
T_p_tab = [380, 410, 420];

CAin_p = 2;
FC_p_tab = [10, 30, 50];

Tin_p = 340;
TCin_p = 310;

xp = cell(3,3);
up = cell(3,3);
zp = cell(3,3);
for i = 1:3
    for j = 1:3
        xp{i,j} = [CA_p; T_p_tab(i)];
        up{i,j} = [CAin_p; FC_p_tab(j)];
        zp{i,j} = [Tin_p; TCin_p];
    end
end

A = cell(3, 3);
B = cell(3, 3);
E = cell(3, 3);
alfa = cell(3, 3);

Ad = cell(3, 3);
Bd = cell(3, 3);
Ed = cell(3, 3);
alfad = cell(3, 3);

tmp1 = zeros(2, 2);

nx = 2;
nu = 2;

N = 50; % prediction horizon
Nu = 5; % control horizon

lambda = 20; % control penalty
Q = kron(eye(N), diag([10000, 1]));
R = lambda * kron(eye(Nu), diag([20, 0.1]));

At = cell(3, 3);
Vt = cell(3, 3);
K1 = cell(3, 3);

for i = 1:3
    for j = 1:3
        T_p = T_p_tab(i);
        FC_p = FC_p_tab(j);
        
        % linearization

        Df_CA_p     =   (ff(CA_p + 1e-8, T_p, CAin_p, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
        Df_T_p      =   (ff(CA_p, T_p + 1e-8, CAin_p, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
        Df_CAin_p   =   (ff(CA_p, T_p, CAin_p + 1e-8, FC_p, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
        Df_FC_p     =   (ff(CA_p, T_p, CAin_p, FC_p + 1e-8, Tin_p, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
        Df_Tin_p    =   (ff(CA_p, T_p, CAin_p, FC_p, Tin_p + 1e-8, TCin_p) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
        Df_TCin_p   =   (ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p + 1e-8) - ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;

        A{i,j} = [Df_CA_p, Df_T_p];
        B{i,j} = [Df_CAin_p, Df_FC_p];
        E{i,j} = [Df_Tin_p, Df_TCin_p];

        alfa{i,j} = ff(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p);

        % discretization using trapezoidal rule

        tmp1 = eye(2) - Ts/2 * A{i,j};
        Ad{i,j} = tmp1 \ (eye(2) + Ts/2 * A{i,j});
        Bd{i,j} = tmp1 \ (Ts * B{i,j});
        Ed{i,j} = tmp1 \ (Ts * E{i,j});
        alfad{i,j} = tmp1 \ (alfa{i,j} * Ts);

        % MPCS

        At{i,j} = zeros(N*nx, nx);
        Mt = zeros(N*nx, Nu*nu);
        Vt{i,j} = zeros(N*nx, nx);

        X = Ad{i,j};
        Y = eye(nx);
        for k=1:N
            At{i,j}((k-1)*nx+1:k*nx, :) = X;
            Vt{i,j}((k-1)*nx+1:k*nx, :) = Y;
            X = X * Ad{i,j};
            Y = Ad{i,j} * Y + eye(nx);
        end

        [nn, ~] = size(Mt);
        for k=1:Nu
            Mt((k-1)*nx+1 : end, (k-1)*nu+1 : k*nu) = Vt{i,j}(1 : nn - (k-1)*nx, :) * Bd{i,j};
        end

        K = (Mt' * Q * Mt + R) \ (Mt' * Q);
        K1{i,j} = K(1:nu, :);
    end
end

% membership functions for fuzzy blending

mf_T = cell(3,1);
mf_T{1} = @(T) trapmf(T, [-1e7, -1e6, 380, 400]);
mf_T{2} = @(T) trimf(T, [380, 400, 420]);
mf_T{3} = @(T) trapmf(T, [400, 420, 1e6, 1e7]);

mf_FC = cell(3,1);
mf_FC{1} = @(FC) trapmf(FC, [-1e7, -1e6, 10, 30]);
mf_FC{2} = @(FC) trimf(FC, [10, 30, 50]);
mf_FC{3} = @(FC) trapmf(FC, [30, 50, 1e6, 1e7]);

mf = cell(3, 3);
for i = 1:3
    for j = 1:3
        mf{i,j} = @(T, FC) min(mf_T{i}(T), mf_FC{j}(FC));
    end
end

whos;

% plot membership functions

% figure;
% T_plot = 370:0.1:430;
% plot(T_plot, mf_T{1}(T_plot), 'r', T_plot, mf_T{2}(T_plot), 'g', T_plot, mf_T{3}(T_plot), 'b');
% xlabel('T'); ylabel('Membership value'); title('Membership functions for T'); grid on; grid minor;

% figure;
% FC_plot = 0:0.1:60;
% plot(FC_plot, mf_FC{1}(FC_plot), 'r', FC_plot, mf_FC{2}(FC_plot), 'g', FC_plot, mf_FC{3}(FC_plot), 'b');
% xlabel('FC'); ylabel('Membership value'); title('Membership functions for FC'); grid on; grid minor;







