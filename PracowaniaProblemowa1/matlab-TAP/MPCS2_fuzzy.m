
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

x2 = 390:1:420;
u2 = 8:0.1:22;
z2 = 290:1:330;

x2n = length(x2);
u2n = length(u2);
z2n = length(z2);

beta1_fun = @(x2) exp(-E_R / x2);
beta2_fun = @(x2,u2,z2) a * u2^b * (z2 - x2) / (u2 + (a * u2^b)/(2 * roc * cpc));

beta1 = zeros(x2n,1);
for ii = 1:x2n
    beta1(ii) = beta1_fun(x2(ii));
end

beta2 = zeros(x2n, u2n, z2n);
for ii = 1:x2n
    for jj = 1:u2n
        for kk = 1:z2n
            beta2(ii,jj,kk) = beta2_fun(x2(ii), u2(jj), z2(kk));
        end
    end
end

beta1_min = min(beta1);
beta1_max = max(beta1);
beta1_avg = (beta1_max + beta1_min) / 2;

beta2_min = min(beta2(:));
beta2_max = max(beta2(:));
beta2_avg = (beta2_max + beta2_min) / 2;


% -----------------------------
mf_beta1_min = @(b) (b <= beta1_min) + (beta1_avg - b) / (beta1_avg - beta1_min) * (b > beta1_min && b < beta1_avg);

mf_beta1_avg = @(b) (b - beta1_min) / (beta1_avg - beta1_min) * (b >= beta1_min && b < beta1_avg) + ...
    (beta1_max - b) / (beta1_max - beta1_avg) * (b >= beta1_avg && b < beta1_max);

mf_beta1_max = @(b) (b - beta1_avg) / (beta1_max - beta1_avg) * (b >= beta1_avg && b < beta1_max) + (b >= beta1_max);

mf_beta2_min = @(b) (b <= beta2_min) + (beta2_avg - b) / (beta2_avg - beta2_min) * (b > beta2_min && b < beta2_avg);

mf_beta2_avg = @(b) (b - beta2_min) / (beta2_avg - beta2_min) * (b >= beta2_min && b < beta2_avg) + ...
    (beta2_max - b) / (beta2_max - beta2_avg) * (b >= beta2_avg && b < beta2_max);

mf_beta2_max = @(b) (b - beta2_avg) / (beta2_max - beta2_avg) * (b >= beta2_avg && b < beta2_max) + (b >= beta2_max);

mf = cell(9, 1);
mf{1} = @(b1,b2) min(mf_beta1_min(b1), mf_beta2_min(b2));
mf{2} = @(b1,b2) min(mf_beta1_min(b1), mf_beta2_avg(b2));
mf{3} = @(b1,b2) min(mf_beta1_min(b1), mf_beta2_max(b2));
mf{4} = @(b1,b2) min(mf_beta1_avg(b1), mf_beta2_min(b2));
mf{5} = @(b1,b2) min(mf_beta1_avg(b1), mf_beta2_avg(b2));
mf{6} = @(b1,b2) min(mf_beta1_avg(b1), mf_beta2_max(b2));
mf{7} = @(b1,b2) min(mf_beta1_max(b1), mf_beta2_min(b2));
mf{8} = @(b1,b2) min(mf_beta1_max(b1), mf_beta2_avg(b2));
mf{9} = @(b1,b2) min(mf_beta1_max(b1), mf_beta2_max(b2));

% -----------------------------
A = cell(9, 1);
B = cell(9, 1);

A{1} = [-F/V - k0 * beta1_min, 0; (h*k0)/(ro*cp) * beta1_min, -F/V];
B{1} = [Fin/V, 0; 0, beta2_min];

A{2} = [-F/V - k0 * beta1_min, 0; (h*k0)/(ro*cp) * beta1_min, -F/V];
B{2} = [Fin/V, 0; 0, beta2_avg];

A{3} = [-F/V - k0 * beta1_min, 0; (h*k0)/(ro*cp) * beta1_min, -F/V];
B{3} = [Fin/V, 0; 0, beta2_max];

A{4} = [-F/V - k0 * beta1_avg, 0; (h*k0)/(ro*cp) * beta1_avg, -F/V];
B{4} = [Fin/V, 0; 0, beta2_min];

A{5} = [-F/V - k0 * beta1_avg, 0; (h*k0)/(ro*cp) * beta1_avg, -F/V];
B{5} = [Fin/V, 0; 0, beta2_avg];

A{6} = [-F/V - k0 * beta1_avg, 0; (h*k0)/(ro*cp) * beta1_avg, -F/V];
B{6} = [Fin/V, 0; 0, beta2_max];

A{7} = [-F/V - k0 * beta1_max, 0; (h*k0)/(ro*cp) * beta1_max, -F/V];
B{7} = [Fin/V, 0; 0, beta2_min];

A{8} = [-F/V - k0 * beta1_max, 0; (h*k0)/(ro*cp) * beta1_max, -F/V];
B{8} = [Fin/V, 0; 0, beta2_avg];

A{9} = [-F/V - k0 * beta1_max, 0; (h*k0)/(ro*cp) * beta1_max, -F/V];
B{9} = [Fin/V, 0; 0, beta2_max];

E = [0, 0; Fin/V, 0];

% -----------------------------
Ad = cell(9, 1);
Bd = cell(9, 1);
Ed = cell(9, 1);

for ii = 1:9
    tmp1 = eye(2) - Ts/2 * A{ii};
    Ad{ii} = tmp1 \ (eye(2) + Ts/2 * A{ii});
    Bd{ii} = tmp1 \ (Ts * B{ii});
    Ed{ii} = tmp1 \ (Ts * E);
end

% -----------------------------
At = cell(9, 1);
Vt = cell(9, 1);
K1 = cell(9, 1);

N = 50; % prediction horizon
Nu = 5; % control horizon
nx = 2; % number of states
nu = 2; % number of controls

for ii = 1:9
    At{ii} = zeros(N*nx, nx);
    Vt{ii} = zeros(N*nx, nx);

    X = Ad{ii};
    Y = eye(nx);
    for i=1:N
        At{ii}((i-1)*nx+1:i*nx, :) = X;
        Vt{ii}((i-1)*nx+1:i*nx, :) = Y;
        X = X * Ad{ii};
        Y = Ad{ii} * Y + eye(nx);
    end

    Mt = zeros(N*nx, Nu*nu);
    [nn, ~] = size(Mt);
    for i=1:Nu
        Mt((i-1)*nx+1 : end, (i-1)*nu+1 : i*nu) = Vt{ii}(1 : nn - (i-1)*nx, :) * Bd{ii};
    end

    lambda = 500; % control penalty
    Q = kron(eye(N), diag([10000, 1]));
    R = lambda * kron(eye(Nu), diag([20, 0.1]));

    K = (Mt' * Q * Mt + R) \ (Mt' * Q);
    K1{ii} = K(1:nu, :);
end






