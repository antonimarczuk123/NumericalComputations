%% __________________________________________________________________
% Przykład uczenia sieci ELM.
% Autor: Antoni Marczuk

clear; clc;


%% __________________________________________________________________
% Przygotowanie danych

% funkcja do aproksymacji
TestFun = @(x) sin(x(1,:)) .* cos(x(2,:));

n_inputs = 2;    % liczba wejść
n_hidden = 5000; % liczba neuronów ukrytych
n_outputs = 1;   % liczba wyjść

fi = @(x) max(0, x); % funkcja aktywacji neuronów ukrytych

n_train = 10000; % liczba próbek treningowych
n_val = 3000;   % liczba próbek walidacyjnych

% Generowanie próbek uczących i walidujących
X_min = 0; X_max = 10;

X_train = X_min + (X_max - X_min) * rand(n_inputs, n_train);
Y_train = TestFun(X_train);
X_train = (X_train - X_min) / (X_max - X_min); % przeskalowanie do [0, 1]

X_val = X_min + (X_max - X_min) * rand(n_inputs, n_val);
Y_val = TestFun(X_val);
X_val = (X_val - X_min) / (X_max - X_min); % przeskalowanie do [0, 1]


%% __________________________________________________________________
% Inicjalizacja wag i biasów dla pierwszej warstwy

% losowe wag pierwszej warstwy
b1 = -5 + 10 * rand(n_hidden, 1);
W1 = -1 + 2 * rand(n_hidden, n_inputs);


%% __________________________________________________________________
% Trenowanie ELM

lambda = 1e-6; % współczynnik regularyzacji

A = [ ones(n_train,1), (fi(b1 + W1 * X_train))' ];
W = (A' * A + lambda * eye(n_hidden + 1)) \ (A' * Y_train');

b2 = W(1);
W2 = W(2:end)';


%% __________________________________________________________________
% Wykresy diagnostyczne

Ymod_train = b2 + W2 * fi(b1 + W1 * X_train);
Ymod_val = b2 + W2 * fi(b1 + W1 * X_val);

MSEtrain = mean((Ymod_train - Y_train).^2);
MSEval = mean((Ymod_val - Y_val).^2);

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


