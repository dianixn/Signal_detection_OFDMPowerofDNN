% Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems
% Block pilot
% Data generation

function [Xtraining_Cell, Xtraining_Array, Ytraining_categorical, Ytraining_regression_cell, Ytraining_regression_array, Xvalidation_Cell, Yvalidation_categorical, Yvalidation_regression] = Data_Generation(Training_set_ratio, SNR, Num_of_frame)

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 63; %126
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Num_of_symbols = 1;
Num_of_pilot = 1;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_interval = Frame_size / Num_of_pilot;
Pilot_starting_location = 1;

length_of_symbol = Num_of_FFT + length_of_CP;

Num_of_QPSK_symbols = Num_of_subcarriers * Num_of_symbols * Num_of_frame;
Num_of_bits = Num_of_QPSK_symbols * log2(M);

Xtraining_Cell = cell(Training_set_ratio * Num_of_frame, 1);
Xtraining_Array = zeros(Num_of_FFT * Frame_size * 2, 1, 1, Training_set_ratio * Num_of_frame);
Ytraining_categorical_double = zeros(Training_set_ratio * Num_of_frame, 1);
Ytraining_regression_cell = cell(Training_set_ratio * Num_of_frame, 1);
Ytraining_regression_array = zeros(Training_set_ratio * Num_of_frame, 16);

Xvalidation_Cell = cell(Num_of_frame - Training_set_ratio * Num_of_frame, 1);
Yvalidation_categorical_double = zeros(Num_of_frame - Training_set_ratio * Num_of_frame, 1);
Yvalidation_regression = cell(Num_of_frame - Training_set_ratio * Num_of_frame, 1);

for Frame = 1 : Num_of_frame

% Data generation
N = Num_of_subcarriers * Num_of_symbols;
data = randi([0 1], N, k);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);

% Pilot inserted
Pilot_value = 1 - 1j;
Pilot_location = Pilot_starting_location : Pilot_interval : Frame_size;

data_location = 1 : Frame_size;
data_location(Pilot_location(:)) = [];

data_in_IFFT = zeros(Num_of_FFT - 1, Frame_size);

data_in_IFFT(:, Pilot_location(:)) = Pilot_value;
data_in_IFFT(:, data_location(:)) = QPSK_signal;
data_in_IFFT = [zeros(1, Frame_size); data_in_IFFT];

% OFDM Transmitter
Transmitted_signal = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);

% Channel

frameLen = size(Transmitted_signal, 1);   % Number of samples to be generated

AA(1) = winner2.AntennaArray('UCA', 16, 0.3);   
AA(2) = winner2.AntennaArray('UCA', 1,  0.05); 

BSIdx    = {1}; % Index in antenna array inventory vector
MSIdx    = [2];     % Index in antenna array inventory vector
numLinks = 1;               % Number of links
range    = 300;             % Layout range (meters)
cfgLayout = winner2.layoutparset(MSIdx, BSIdx, numLinks, AA, range);

cfgLayout.Pairing = [1; 2];  % Index in cfgLayout.Stations
cfgLayout.ScenarioVector = [11];     % 6 for B4, 11 for C2 and 13 for C4
cfgLayout.PropagConditionVector = [0];  % 0 for NLOS

% Define Position

% Number of BS sectors and MSs in the system
numBSSect = sum(cfgLayout.NofSect);
numMS = length(MSIdx);

% Set up positions for BS sectors. Same position for the third, fourth and
% fifth sectors as they belong to one BS.
cfgLayout.Stations(1).Pos(1:2) = [150;  150]; 

% Set up MS positions
cfgLayout.Stations(2).Pos(1:2)  = [10;  180];  % 50m from 1st BS

% Randomly draw MS velocity
for i = numBSSect + (1:numMS)
    cfgLayout.Stations(i).Velocity = rand(3,1) - 0.5;
end

% Configure Model Parameters

cfgWim = winner2.wimparset;
cfgWim.NumTimeSamples      = frameLen;
cfgWim.IntraClusterDsUsed  = 'yes';
cfgWim.CenterFrequency     = 5.25e9;
cfgWim.UniformTimeSampling = 'no';
cfgWim.ShadowingModelUsed  = 'yes';
cfgWim.PathLossModelUsed   = 'yes';
cfgWim.RandomSeed          = 31415926;  % For repeatability

% Create WINNER II Channel System Object(TM)

WINNERChan = comm.WINNER2Channel(cfgWim, cfgLayout);
chanInfo = info(WINNERChan);

txSig = cellfun(@(x) ones(1,x) .* Transmitted_signal, ...
    num2cell(chanInfo.NumBSElements)', 'UniformOutput', false);

Multitap_Channel_Signal = WINNERChan(txSig);

%AWGN
Data_power = sum(power(abs(QPSK_symbol), 2)) / (length(QPSK_symbol));

SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
SNR_OFDM_HEX = 10 ^ (SNR_OFDM / 10);
Noise_power = Data_power / SNR_OFDM_HEX;

Nvariance = sqrt(Noise_power/2); % QPSK has two paths of signal
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

Multitap_Channel_Signal = Multitap_Channel_Signal{1} + n;

% OFDM Receiver

Channel_signal_when_h_is_known = [1; zeros(size(Multitap_Channel_Signal, 1) - 1, 1)];
[Received_data, ~] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);

[feature_of_OFDM_frame, label_of_OFDM_frame, label_of_regression] = Extract_Feature_OFDM(Received_data, dataSym(1:2), M, QPSK_signal(1:8));

if Frame <= fix(Training_set_ratio * Num_of_frame)
    Xtraining_Cell{Frame, 1} = feature_of_OFDM_frame;
    Xtraining_Array(:, :, 1, Frame) = feature_of_OFDM_frame;
    Ytraining_categorical_double(Frame, 1) = label_of_OFDM_frame;
    Ytraining_regression_cell{Frame, 1} = label_of_regression;
    Ytraining_regression_array(Frame, :) = label_of_regression;
else
    Xvalidation_Cell{Frame - Training_set_ratio * Num_of_frame, 1} = feature_of_OFDM_frame;
    Yvalidation_categorical_double(Frame - Training_set_ratio * Num_of_frame, 1) = label_of_OFDM_frame;
    Yvalidation_regression{Frame - Training_set_ratio * Num_of_frame, 1} = label_of_regression;
end

end

Ytraining_categorical = categorical(Ytraining_categorical_double);
Yvalidation_categorical = categorical(Yvalidation_categorical_double);

end
