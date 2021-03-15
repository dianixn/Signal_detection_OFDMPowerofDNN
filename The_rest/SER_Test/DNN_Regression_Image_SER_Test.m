[~, XTrain, ~, ~, Ytraining_regression, ~, ~, ~] = Data_Generation(0.95, 30, 75000);
Training_Set_Rate = 0.95;
% Set up layers
Layers = [
    imageInputLayer([256 1 1],"Name","imageinput","Normalization","none")
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
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'L2Regularization',0.003, ...
    'Plots','training-progress');

% Train Network
[DNN_Trained, info] = trainNetwork(XTrain, Ytraining_regression, Layers, Options);

% Test Network
SNR_Test = 5;

[~, XTrain_test, ~, ~, Ytraining_regression_test, ~, ~, ~] = Data_Generation(1, SNR_Test, 10000);
Ypred_test = predict(DNN_Trained, XTrain_test);

Ytraining_regression_Test = Ytraining_regression_test';%transpose(Ytraining_regression_test)
Ypred_Test = transpose(Ypred_test);

Label_symbol = Ytraining_regression_Test(1:2:end, :) + 1j * Ytraining_regression_Test(2:2:end, :);
Label_dataSym = QPSK_Demodulator(Label_symbol);

Predicted_symbols = Ypred_Test(1:2:end, :) + 1j * Ypred_Test(2:2:end, :);
Predicted_dataSym = QPSK_Demodulator(Predicted_symbols);

SER = sum(Predicted_dataSym ~= Label_dataSym) / length(Label_dataSym);
