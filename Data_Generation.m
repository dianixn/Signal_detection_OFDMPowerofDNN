% Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems
% Block pilot
% Data generation

function [Xtraining_Cell, Xtraining_Array, Ytraining_regression_array, Xvalidation_regression, Yvalidation_regression] = Data_Generation(Training_set_ratio, SNR, Num_of_frame)

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

Xtraining_Cell = cell(Training_set_ratio * Num_of_frame, 1);
Xtraining_Array = zeros(Num_of_FFT * Frame_size * 2, 1, 1, Training_set_ratio * Num_of_frame);
Ytraining_regression_array = zeros(1, 1, 16, Training_set_ratio * Num_of_frame);

Xvalidation_regression = zeros(Num_of_FFT * Frame_size * 2, 1, 1, Num_of_frame - Training_set_ratio * Num_of_frame);
Yvalidation_regression = zeros(1, 1, 16, Num_of_frame - Training_set_ratio * Num_of_frame);

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

% AWGN Channel
Data_power = sum(power(abs(QPSK_symbol), 2)) / (length(QPSK_symbol));

SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
SNR_OFDM_HEX = 10 ^ (SNR_OFDM / 10);
Noise_power = Data_power / SNR_OFDM_HEX;

Nvariance = sqrt(Noise_power/2); % QPSK has two paths of signal
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

% Multipath Rayleigh Fading Channel
Multitap_h = [(randn + 1j * randn);...
    (randn + 1j * randn) / 2;...
    (randn + 1j * randn) / 4;...
    (randn + 1j * randn) / 8;...
    (randn + 1j * randn) / 16] / (1.9375 * sqrt(pi/2));

% linear convolution
Multitap_Channel_Signal = conv(Transmitted_signal, Multitap_h);

% circular convolution
%Multitap_Channel_Signal = cconv(Transmitted_signal, Multitap_h, length(Transmitted_signal));

Multitap_Channel_Signal = Multitap_Channel_Signal(1 : length(Transmitted_signal)) + n;


% OFDM Receiver
Channel_signal_when_h_is_known = ones(160,1);
[Received_data, ~] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);

[feature_of_OFDM_frame, ~, label_of_regression] = Extract_Feature_OFDM(Received_data, dataSym(1:2), M, QPSK_signal(1:8));

if Frame <= fix(Training_set_ratio * Num_of_frame)
    Xtraining_Cell{Frame, 1} = feature_of_OFDM_frame;
    Xtraining_Array(:, 1, 1, Frame) = feature_of_OFDM_frame;
    Ytraining_regression_array(1, 1, :, Frame) = label_of_regression;
else
    Xvalidation_regression(:, 1, 1, Frame - Training_set_ratio * Num_of_frame, 1) = feature_of_OFDM_frame;
    Yvalidation_regression(1, 1, :, Frame - Training_set_ratio * Num_of_frame) = label_of_regression;
end

end

end
