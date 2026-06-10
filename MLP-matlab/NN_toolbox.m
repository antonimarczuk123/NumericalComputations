%% __________________________________________________________________
% Przykład uczenia sieci z wykorzystaniem toolboxu Neural Network.
% Autor: Antoni Marczuk

clear; clc;


%% __________________________________________________________________
% Przygotowanie danych

% funkcja do aproksymacji
TestFun = @(x) sin(x(1,:)) .* cos(x(2,:));

inputSize = 2; % liczba wejść
N = 15000; % liczba próbek

% generowanie próbek
Xmin = 0;
Xmax = 10;
X = Xmin + (Xmax - Xmin) * rand(inputSize, N); 
Y = TestFun(X);
X = 2 * (X - Xmin) / (Xmax - Xmin) - 1; % skalowanie do [-1, 1]


%% __________________________________________________________________
% Konfiguracja i trenowanie sieci

K=[25, 25, 25]; % liczba neuronów w warstwach ukrytych
net = feedforwardnet(K); % tworzenie sieci

% wybór algorytmu optymalizującego
net.trainFcn = 'trainlm'; 
%'trainlm': Levenberg-Marquardt
%'trainbr': Bayesian Regularization
%'trainbfg': BFGS Quasi-Newton
%'traincgf': Fletcher-Powell Conjugate Gradient
%'traincgp': Polak-Ribiére Conjugate Gradient
%'traingdx': Variable Learning Rate Gradient Descent
%'traingdm': Gradient Descent with Momentum
%'traingd': Gradient Descent

net.performFcn = 'mse'; % funkcja błędu
net.trainParam.epochs = 3000; % maksymalna liczba epok
net.trainParam.goal = 1e-6; % cel treningu (minimalny błąd)
net.trainParam.max_fail = 20; % early stopping: maksymalna liczba epok bez poprawy błędu walidującego
net.trainParam.showWindow = false; % wyświetlanie okno śledzęce postęp
net.trainParam.showCommandLine = true; % wypisuj informacje o postępie treningu w konsoli
net.trainParam.show=10; % częstotliwość wyświetlania informacji o postępie treningu
net.divideFcn = 'dividerand';   % losowy podział danych
net.divideParam.trainRatio = 0.7; % 70% danych uczących
net.divideParam.valRatio   = 0.3; % 30% danych weryfikujących
net.divideParam.testRatio  = 0; % 0% danych testowych

net.inputs{1}.processFcns  = {}; % wyłączenie przetwarzania wejść (m.in. skalowania)
net.outputs{1}.processFcns = {}; % wyłączenie przetwarzania wyjść (m.in. skalowania)

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
% tanh - 'tansig'
% ReLU - 'poslin'
% lin - 'purelin'

[net, tr] = train(net, X, Y, 'useParallel', 'no', 'useGPU', 'yes');

% --- Wykres MSE w czasie treningu ---

figure;

subplot(2,1,1);
semilogy(tr.epoch, tr.perf);
grid on;
grid minor;
xlabel('Epoch');
ylabel('MSE');
title('Training MSE over Epochs');

subplot(2,1,2);
semilogy(tr.epoch, tr.vperf);
grid on;
grid minor;
xlabel('Epoch');
ylabel('MSE');
title('Validation MSE over Epochs');


%% __________________________________________________________________
% Pobranie wyznaczonych wag

b1 = net.b{1};
b2 = net.b{2};
b3 = net.b{3};
b4 = net.b{4};
W1 = net.IW{1,1};   % input -> layer 1
W2 = net.LW{2,1};   % layer 1 -> layer 2
W3 = net.LW{3,2};   % layer 2 -> layer 3
W4 = net.LW{4,3};   % layer 3 -> output layer


%% __________________________________________________________________
% Ocena modelu

% --- Ocena modelu na zbiorze uczącym i walidującym ---

X_train = X(:, tr.trainInd); 
Y_train=Y(:, tr.trainInd);
% Ymod_train = net(Xtrain);
Ymod_train = b4 + W4 * (tansig(b3 + W3 * (tansig(b2 + W2 * (tansig(b1 + W1 * X_train))))));

X_val = X(:, tr.valInd); 
Y_val=Y(:, tr.valInd);
% Ymod_val = net(Xval);
Ymod_val = b4 + W4 * (tansig(b3 + W3 * (tansig(b2 + W2 * (tansig(b1 + W1 * X_val))))));

MSEtrain = mse(Y_train, Ymod_train);
MSEval = mse(Y_val, Ymod_val);

var_norm_MSE_train = MSEtrain / var(Y_train);
var_norm_MSE_val = MSEval / var(Y_val);

fprintf('(train) MSE = %e,  var-norm-MSE = %e\n', MSEtrain, var_norm_MSE_train);
fprintf('(val) MSE = %e,  var-norm-MSE = %e\n', MSEval, var_norm_MSE_val);

% --- Parity Plot ---

figure;

subplot(1,2,1);
hold on;
scatter(Y_train, Ymod_train, 3, 'b', 'filled');
plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], 'r--', 'LineWidth', 1.5);
grid on;
grid minor;
xlabel('True Y');
ylabel('Predicted Y');
title('Model Prediction vs True Values (Training Set)');

subplot(1,2,2);
hold on;
scatter(Y_val, Ymod_val, 3, 'b', 'filled');
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--', 'LineWidth', 1.5);
grid on;
grid minor;
xlabel('True Y');
ylabel('Predicted Y');
title('Model Prediction vs True Values (Validation Set)');

% --- Residual Plot ---

figure;

subplot(1,2,1);
hold on;
scatter(Y_train, Ymod_train - Y_train, 3, 'b', 'filled');
plot([min(Ymod_train), max(Ymod_train)], [0, 0], 'r--', 'LineWidth', 1.5);
grid on;
grid minor;
xlabel('True Y');
ylabel('Residuals = Predicted Y - True Y');
title('Residual Plot (Training Set)');

subplot(1,2,2);
hold on;
scatter(Y_val, Ymod_val - Y_val, 3, 'b', 'filled');
plot([min(Ymod_val), max(Ymod_val)], [0, 0], 'r--', 'LineWidth', 1.5);
grid on;
grid minor;
xlabel('True Y');
ylabel('Residuals = Predicted Y - True Y');
title('Residual Plot (Validation Set)');


