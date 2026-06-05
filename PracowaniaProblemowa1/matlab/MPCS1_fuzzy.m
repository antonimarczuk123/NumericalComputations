%% ================================================================================
% Run this file first to compute the MPCS matrices. 
% Then run MPCS2analytic.m or MPCS2numeric.m to simulate the closed-loop system.

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

% Controlled variables: x = [CA; T]
% Manipulated variables: u = [CAin; FC]
% Disturbance variables: z = [Tin; TCin]

% Object equations
% dCA/dt = f1(CA, T, CAin, FC, Tin, TCin)
% dT/dt = f2(CA, T, CAin, FC, Tin, TCin)
f = @(CA, T, CAin, FC, Tin, TCin) [
    (1/V) * (Fin*CAin - F*CA - V * k0 * exp(-E_R/T) * CA);
    (1/(V*ro*cp)) * (Fin*ro*cp*Tin - F*ro*cp*T + V*h*k0*exp(-E_R/T)*CA - (a * FC^(b+1) / (FC + (a * FC^b) / (2 * roc * cpc))) * (T - TCin))
];

% Operating point (steady state)
CAin_p = 2; % kmol/m^3
Tin_p = 343; % K
TCin_p = 310; % K

FC_p_tab = [7, 9, 12, 15, 17, 18]; % m^3/min
CA_p_tab = [0.0348784655658222, 0.05557088024052338, 0.09821109463204293, 0.1597172021766315, 0.2169364570631375, 1.828622864494349]; % kmol/m^3
T_p_tab = [438.5550598263770, 427.8255516809578, 415.2091350393931, 404.7356680017397, 398.2004016808073, 328.0431883372068]; % K

Adf = cell(6, 1);
Bdf = cell(6, 1);
Edf = cell(6, 1);
Atf = cell(6, 1);
Vtf = cell(6, 1);
K1f = cell(6, 1);

for iif = 1:6
    FC_p = FC_p_tab(iif);
    CA_p = CA_p_tab(iif);
    T_p = T_p_tab(iif);

    % ================================================================================
    % Linearization
    % [dCA/dt; dT/dt] = A*[CA - CA_p; T - T_p] + B*[CAin - CAin_p; FC - FC_p] + E*[Tin - Tin_p; TCin - TCin_p]

    Df_CA_p     =   (f(CA_p + 1e-8, T_p, CAin_p, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_T_p      =   (f(CA_p, T_p + 1e-8, CAin_p, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_CAin_p   =   (f(CA_p, T_p, CAin_p + 1e-8, FC_p, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_FC_p     =   (f(CA_p, T_p, CAin_p, FC_p + 1e-8, Tin_p, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_Tin_p    =   (f(CA_p, T_p, CAin_p, FC_p, Tin_p + 1e-8, TCin_p) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;
    Df_TCin_p   =   (f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p + 1e-8) - f(CA_p, T_p, CAin_p, FC_p, Tin_p, TCin_p)) / 1e-8;

    A = [Df_CA_p, Df_T_p];
    B = [Df_CAin_p, Df_FC_p];
    E = [Df_Tin_p, Df_TCin_p];


    % ================================================================================
    % Find discretization of linearized model with RK4

    Ts = 0.05; % sampling time [min]

    % d(x-xp)/dt = A*(x-xp) + B*(u-up) + E*(z-zp)
    % x(k+1)-xp = Ad*(x(k)-xp) + Bd*(u(k)-up) + Ed*(z(k)-zp)

    % where Ad, Bd, Ed:
    % Ad = eye(size(A)) + A*Ts + (A^2)*(Ts^2)/2 + (A^3)*(Ts^3)/6 + (A^4)*(Ts^4)/24;
    % Bd = B*Ts + A*B*(Ts^2)/2 + A^2*B*(Ts^3)/6 + A^3*B*(Ts^4)/24;
    % Ed = E*Ts + A*E*(Ts^2)/2 + A^2*E*(Ts^3)/6 + A^3*E*(Ts^4)/24;

    A2 = A*A;
    A3 = A2*A;
    Tmp = Ts*eye(2) + A*(Ts^2)/2 + A2*(Ts^3)/6 + A3*(Ts^4)/24;
    Ad = eye(2) + A*Tmp;
    Bd = Tmp*B;
    Ed = Tmp*E;

    Adf{iif} = Ad;
    Bdf{iif} = Bd;
    Edf{iif} = Ed;

    % ===============================================================================
    % MPCS matrices

    N = 50; % prediction horizon (2.5 min)
    Nu = 5; % control horizon

    [nx, nu]=size(Bd);

    At = zeros(N*nx, nx);
    Mt = zeros(N*nx, Nu*nu);
    Vt = zeros(N*nx, nx);

    X = Ad;
    Y = eye(nx);
    for i=1:N
        At((i-1)*nx+1:i*nx, :) = X;
        Vt((i-1)*nx+1:i*nx, :) = Y;
        X = X * Ad;
        Y = Ad * Y + eye(nx);
    end

    [nn, ~] = size(Mt);
    for i=1:Nu
        Mt((i-1)*nx+1 : end, (i-1)*nu+1 : i*nu) = Vt(1 : nn - (i-1)*nx, :) * Bd;
    end

    lambda = 20; % control penalty
    Q = kron(eye(N), diag([10000, 1]));
    R = lambda * kron(eye(Nu), diag([20, 0.1]));

    K = (Mt' * Q * Mt + R) \ (Mt' * Q);
    K1 = K(1:nu, :);

    Atf{iif} = At;
    Vtf{iif} = Vt;
    K1f{iif} = K1;

end

mf = cell(6, 1);
mf{1} = @(FC) trapmf(FC, [-Inf, Inf, 7, 9]);
mf{2} = @(FC) trimf(FC, [7, 9, 12]);
mf{3} = @(FC) trimf(FC, [9, 12, 15]);
mf{4} = @(FC) trimf(FC, [12, 15, 17]);
mf{5} = @(FC) trimf(FC, [15, 17, 18]);
mf{6} = @(FC) trapmf(FC, [17, 18, Inf, Inf]);

% ===============================================================================
% clear temporary variables
clear i iif nn nx;
clear X Y;
clear K K1 lambda;
clear Ad At Bd Ed Vt;
clear Q R Mt;
clear CA_p FC_p T_p;
clear J;
clear A B E;
clear Df_CA_p Df_T_p Df_CAin_p Df_FC_p Df_Tin_p Df_TCin_p;
clear A2 A3 Tmp;
clear f;
whos;

% We are ready to run simulation in MPCS2analytic.m





