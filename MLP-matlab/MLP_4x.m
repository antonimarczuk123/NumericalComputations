%% __________________________________________________________________
% Przykład uczenia sieci z czterema warstwami ukrytymi.
% Autor: Antoni Marczuk

clear; clc;


%% __________________________________________________________________
% Przygotowanie danych

% funkcja do aproksymacji
TestFun = @(x) sin(x(1,:)) + cos(x(2,:));

n_inputs = 2;    % liczba wejść
n_hidden = [30, 30, 30, 30]; % liczba neuronów w warstwach ukrytych
n_outputs = 1;   % liczba wyjść

fi = @(x) tanh(x); % funkcja aktywacji neuronów ukrytych
dfi = @(x) 1 - tanh(x).^2; % pochodna funkcji aktywacji

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
% Inicjalizacja wag i biasów sieci. 


% Losowa inicjalizacja wag i biasów

b1 = rand(n_hidden(1), 1) - 0.5;
W1 = rand(n_hidden(1), n_inputs) - 0.5;

b2 = rand(n_hidden(2), 1) - 0.5;
W2 = rand(n_hidden(2), n_hidden(1)) - 0.5;

b3 = rand(n_hidden(3), 1) - 0.5;
W3 = rand(n_hidden(3), n_hidden(2)) - 0.5;

b4 = rand(n_hidden(4), 1) - 0.5;
W4 = rand(n_hidden(4), n_hidden(3)) - 0.5;

b5 = rand(n_outputs, 1) - 0.5;
W5 = rand(n_outputs, n_hidden(4)) - 0.5;

% zerowa inicjalizacja poprzednich kroków minimalizacji

p_b1_old = zeros(size(b1));
p_W1_old = zeros(size(W1));

p_b2_old = zeros(size(b2));
p_W2_old = zeros(size(W2));

p_b3_old = zeros(size(b3));
p_W3_old = zeros(size(W3));

p_b4_old = zeros(size(b4));
p_W4_old = zeros(size(W4));

p_b5_old = zeros(size(b5));
p_W5_old = zeros(size(W5));


%% __________________________________________________________________
% Uczenie sieci metodą Nesterov SGD.


max_epochs = 3000;
learning_rate = 0.001;
momentum = 0.9;
batch_size = 64;


% deklaracja potrzebnych tablic 
% (dzięki wcześniejszej deklaracji pętla ucząca działa szybciej)

idx = zeros(1, batch_size);
X = zeros(n_inputs, batch_size);
Y = zeros(n_outputs, batch_size);

% ---

Z1 = zeros(n_hidden(1), batch_size);
V1 = zeros(n_hidden(1), batch_size);

Z2 = zeros(n_hidden(2), batch_size);
V2 = zeros(n_hidden(2), batch_size);

Z3 = zeros(n_hidden(3), batch_size);
V3 = zeros(n_hidden(3), batch_size);

Z4 = zeros(n_hidden(4), batch_size);
V4 = zeros(n_hidden(4), batch_size);

Z5 = zeros(n_outputs, batch_size);
Ymod = zeros(n_outputs, batch_size);

% ---

p_b1 = zeros(size(b1));
p_W1 = zeros(size(W1));

p_b2 = zeros(size(b2));
p_W2 = zeros(size(W2));

p_b3 = zeros(size(b3));
p_W3 = zeros(size(W3));

p_b4 = zeros(size(b4));
p_W4 = zeros(size(W4));

p_b5 = zeros(size(b5));
p_W5 = zeros(size(W5));

% ---

dL5 = zeros(n_outputs, batch_size);
dL4 = zeros(n_hidden(4), batch_size);
dL3 = zeros(n_hidden(3), batch_size);
dL2 = zeros(n_hidden(2), batch_size);
dL1 = zeros(n_hidden(1), batch_size);

% ---

dE_db1 = zeros(size(b1));
dE_dW1 = zeros(size(W1));

dE_db2 = zeros(size(b2));
dE_dW2 = zeros(size(W2));

dE_db3 = zeros(size(b3));
dE_dW3 = zeros(size(W3));

dE_db4 = zeros(size(b4));
dE_dW4 = zeros(size(W4));

dE_db5 = zeros(size(b5));
dE_dW5 = zeros(size(W5));

% ---

b1_look = zeros(size(b1));
W1_look = zeros(size(W1));

b2_look = zeros(size(b2));
W2_look = zeros(size(W2));

b3_look = zeros(size(b3));
W3_look = zeros(size(W3));

b4_look = zeros(size(b4));
W4_look = zeros(size(W4));

b5_look = zeros(size(b5));
W5_look = zeros(size(W5));

% ---

MSEtrainTable = zeros(1, max_epochs+1);
MSEvalTable = zeros(1, max_epochs+1);


% Wyjściowe wartości MSE

Ymod_train = b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_train))));
Ymod_val =   b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_val))));

MSEtrain = mean((Ymod_train - Y_train).^2);
MSEval =   mean((Ymod_val - Y_val).^2);

MSEtrainTable(1) = MSEtrain;
MSEvalTable(1) = MSEval;


% uczenie sieci
msg = sprintf('Epoch [%d/%d]  MSE train: %.8e,  MSE val: %.8e', i, max_epochs, MSEtrain, MSEval);
h = waitbar(0, msg);
for i = 1:max_epochs
    for j = 1:100
        idx = randperm(n_train, batch_size);
        X = X_train(:, idx);
        Y = Y_train(:, idx);

        %  Nesterov lookahead
        b1_look = b1 + momentum * p_b1_old;
        W1_look = W1 + momentum * p_W1_old;    
        b2_look = b2 + momentum * p_b2_old;
        W2_look = W2 + momentum * p_W2_old;    
        b3_look = b3 + momentum * p_b3_old;
        W3_look = W3 + momentum * p_W3_old;    
        b4_look = b4 + momentum * p_b4_old;
        W4_look = W4 + momentum * p_W4_old;    
        b5_look = b5 + momentum * p_b5_old;
        W5_look = W5 + momentum * p_W5_old;
        
        % Forward pass
        Z1 = b1_look + W1_look * X;
        V1 = fi(Z1);
        Z2 = b2_look + W2_look * V1;
        V2 = fi(Z2);
        Z3 = b3_look + W3_look * V2;
        V3 = fi(Z3);
        Z4 = b4_look + W4_look * V3;
        V4 = fi(Z4);
        Z5 = b5_look + W5_look * V4;
        Ymod = Z5;

        % Backward pass
        dL5 = 2 * (Ymod - Y);
        dL4 = (W5_look' * dL5) .* dfi(Z4);
        dL3 = (W4_look' * dL4) .* dfi(Z3);
        dL2 = (W3_look' * dL3) .* dfi(Z2);
        dL1 = (W2_look' * dL2) .* dfi(Z1);

        % Gradienty
        dE_db5 = mean(dL5, 2);
        dE_dW5 = (dL5 * V4') / batch_size;
        dE_db4 = mean(dL4, 2);
        dE_dW4 = (dL4 * V3') / batch_size;
        dE_db3 = mean(dL3, 2);
        dE_dW3 = (dL3 * V2') / batch_size;
        dE_db2 = mean(dL2, 2);
        dE_dW2 = (dL2 * V1') / batch_size;
        dE_db1 = mean(dL1, 2);
        dE_dW1 = (dL1 * X') / batch_size;

        % Aktualizacja kroków minimalizacji
        p_b5 = momentum * p_b5_old - learning_rate * dE_db5;
        p_W5 = momentum * p_W5_old - learning_rate * dE_dW5;
        p_b4 = momentum * p_b4_old - learning_rate * dE_db4;
        p_W4 = momentum * p_W4_old - learning_rate * dE_dW4;
        p_b3 = momentum * p_b3_old - learning_rate * dE_db3;
        p_W3 = momentum * p_W3_old - learning_rate * dE_dW3;
        p_b2 = momentum * p_b2_old - learning_rate * dE_db2;
        p_W2 = momentum * p_W2_old - learning_rate * dE_dW2;
        p_b1 = momentum * p_b1_old - learning_rate * dE_db1;
        p_W1 = momentum * p_W1_old - learning_rate * dE_dW1;

        % Aktualizacja wag i biasów
        b5 = b5 + p_b5;
        W5 = W5 + p_W5;
        b4 = b4 + p_b4;
        W4 = W4 + p_W4;
        b3 = b3 + p_b3;
        W3 = W3 + p_W3;
        b2 = b2 + p_b2;
        W2 = W2 + p_W2;
        b1 = b1 + p_b1;
        W1 = W1 + p_W1;

        % Zapamiętanie poprzednich kroków
        p_b5_old = p_b5;
        p_W5_old = p_W5;
        p_b4_old = p_b4;
        p_W4_old = p_W4;
        p_b3_old = p_b3;
        p_W3_old = p_W3;
        p_b2_old = p_b2;
        p_W2_old = p_W2;
        p_b1_old = p_b1;
        p_W1_old = p_W1;
    end

    % Obliczenie MSE na zbiorze uczącym i walidującym
    Ymod_train = b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_train))));
    Ymod_val =   b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_val))));

    MSEtrain = mean((Ymod_train - Y_train).^2);
    MSEval =   mean((Ymod_val - Y_val).^2);

    MSEtrainTable(i+1) = MSEtrain;
    MSEvalTable(i+1) = MSEval;

    msg = sprintf('Epoch [%d/%d]  MSE train: %.8e,  MSE val: %.8e', i, max_epochs, MSEtrain, MSEval);
    waitbar(i / max_epochs, h, msg);
end
close(h)

% Wyświetlanie przebiegu uczenia

figure;

subplot(2,1,1);
semilogy(0:max_epochs, MSEtrainTable);
grid on;
grid minor;
xlabel('Epoch');
ylabel('MSE');
title('Training MSE over Epochs');

subplot(2,1,2);
semilogy(0:max_epochs, MSEvalTable);
grid on;
grid minor;
xlabel('Epoch');
ylabel('MSE');
title('Validation MSE over Epochs');


%% __________________________________________________________________
% Wykresy diagnostyczne

% --- Ocena modelu ---

Ymod_train = b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_train))));
Ymod_val =   b5 + W5 * fi(b4 + W4 * fi(b3 + W3 * fi(b2 + W2 * fi(b1 + W1 * X_val))));

MSEtrain = mean((Ymod_train - Y_train).^2);
MSEval =   mean((Ymod_val - Y_val).^2);

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


