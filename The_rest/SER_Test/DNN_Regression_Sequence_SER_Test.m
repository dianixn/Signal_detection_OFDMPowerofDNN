[XTrain, ~, ~, Ytraining_regression, ~, XValidation, ~, Yvalidation_regression] = Data_Generation(0.95, 30, 75000);
Training_Set_Rate = 0.95;
% Set up layers
Layers = [
    sequenceInputLayer(256,"Name","sequence")
    fullyConnectedLayer(500,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(250,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(120,"Name","fc_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(16,"Name","fc_4")
    regressionLayer("Name","regressionoutput")];

% Option settings
ValidationFrequency = ceil(1 / (1 - Training_Set_Rate));
%3 2 % 0.7 0.8
Options = trainingOptions('rmsprop', ...
    'MaxEpochs',13, ...
    'MiniBatchSize',10, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'LearnRateDropPeriod',2, ...
    'ValidationData',{XValidation,Yvalidation_regression}, ...
    'ValidationFrequency',ValidationFrequency, ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'L2Regularization',0.003, ...
    'Plots','training-progress');

% Train Network
[DNN_Trained, info] = trainNetwork(XTrain, Ytraining_regression, Layers, Options);

% Test Network
SNR_Test = 5;

[XTrain_test, ~, ~, Ytraining_regression_test, ~, ~, ~, ~] = Data_Generation(1, SNR_Test, 10000);

Ypred_test = predict(DNN_Trained, XTrain_test);

Ypred_test = cellfun(@(x) x(:),Ypred_test,'UniformOutput',false);
Ytraining_regression_test = cellfun(@(x) x(:),Ytraining_regression_test,'UniformOutput',false);

Sybmol_label = zeros(16, size(Ytraining_regression_test, 1));
for i = 1 : size(Ytraining_regression_test, 1)
    Sybmol_label(:, i) = cell2mat(Ytraining_regression_test(i, 1));
end
    Label_symbol = Sybmol_label(1:2:end, :) + 1j * Sybmol_label(2:2:end, :);
    Label_dataSym = QPSK_Demodulator(Label_symbol);
    
Sybmol_Pred = zeros(16, size(Ytraining_regression_test, 1));
for i = 1 : size(Ytraining_regression_test, 1)
    Sybmol_Pred(:, i) = cell2mat(Ypred_test(i, 1));
end
    Predicted_symbols = Sybmol_Pred(1:2:end, :) + 1j * Sybmol_Pred(2:2:end, :);
    Predicted_dataSym = QPSK_Demodulator(Predicted_symbols);

SER = sum(Predicted_dataSym ~= Label_dataSym) / length(Label_dataSym);
