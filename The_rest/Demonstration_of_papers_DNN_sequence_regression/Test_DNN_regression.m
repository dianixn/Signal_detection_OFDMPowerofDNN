% Test DNN performance

Training_Set_Rate = 0.95;
SNR = 30;
Num_of_training_and_validation_frame = 75000;

[XTrain, ~, ~, YTrain, ~, XValidation, ~, YValidation] = Data_Generation(Training_Set_Rate, SNR, Num_of_training_and_validation_frame);
[DNN_Trained, info] = Train_DNN(XTrain, YTrain, XValidation, YValidation, Training_Set_Rate);

SNR_Range = 5:25;
DNN_BER_over_SNR = zeros(length(SNR_Range), 1);
DNN_SER_over_SNR = zeros(length(SNR_Range), 1);

for SNR = SNR_Range

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 63; %126
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Num_of_frame = 10000;
Num_of_symbols = 1;
Num_of_pilot = 1;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_interval = Frame_size / Num_of_pilot;
Pilot_starting_location = 1;

length_of_symbol = Num_of_FFT + length_of_CP;

Num_of_QPSK_symbols_DNN = 8 * Num_of_symbols * Num_of_frame;
Num_of_bits_DNN = Num_of_QPSK_symbols_DNN * k;

DNN_numErrs_sym = zeros(Num_of_frame, 1);
DNN_SER_in_frame = zeros(Num_of_frame, 1);
DNN_numErrs_bit = zeros(Num_of_frame, 1);
DNN_BER_in_frame = zeros(Num_of_frame, 1);

for Frame = 1:Num_of_frame

% Data generation
N = Num_of_subcarriers * Num_of_symbols;
data = randi([0 1], N, k);
Data = reshape(data, [], 1);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);

% Pilot inserted
Pilot_value = 1 - 1j;
Pilot_location = Pilot_starting_location : Pilot_interval : Frame_size;
Num_of_Pilot_per_Frame = length(Pilot_location) * Num_of_subcarriers;

data_location = 1 : Frame_size;
data_location(Pilot_location(:)) = [];

data_in_IFFT = zeros(Num_of_FFT - 1, Num_of_symbols);

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
AWGN_Signal = Transmitted_signal + n;

% Multipath Rayleigh Fading Channel
Multitap_h = [(randn + 1j * randn);...
    (randn + 1j * randn) / 2;...
    (randn + 1j * randn) / 4;...
    (randn + 1j * randn) / 8;...
    (randn + 1j * randn) / 16] / (sqrt(2) * 1.9375);

% linear convolution
Multitap_Channel_Signal = conv(Transmitted_signal, Multitap_h);
Multitap_Channel_Signal = Multitap_Channel_Signal(1 : length(Transmitted_signal)) + n;



% OFDM Receiver
[Unrecovered_signal, Unrecovered_signal_when_h_is_known] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);

% Channel estimation

% Deep learning
% Detect 8 QPSK symbols only
[DNN_Unrecovered_signal, ~, ~] = Extract_Feature_OFDM(Unrecovered_signal,dataSym, QPSK_signal(1:8));
Received_data_DNN = predict(DNN_Trained, DNN_Unrecovered_signal);

DNN_Received_data = Received_data_DNN(1:2:end, :) + 1j * Received_data_DNN(2:2:end, :);

%scatterplot(Received_data(:, 1))

% QPSK demodulator

DNN_dataSym_Rx = QPSK_Demodulator(DNN_Received_data);

DNN_dataSym_Received = de2bi(DNN_dataSym_Rx, 2);
DNN_Data_Received = reshape(DNN_dataSym_Received, [], 1);

% BER calculation in each frame
DNN_numErrs_sym(Frame, 1) = sum(sum(round(dataSym(1:8)) ~= round(DNN_dataSym_Rx)));
DNN_SER_in_frame(Frame, 1) = DNN_numErrs_sym(Frame, 1) / length(DNN_dataSym_Rx);

DNN_numErrs_bit(Frame, 1) = sum(sum(round(reshape(de2bi(dataSym(1:8), 2),[],1)) ~= round(DNN_Data_Received)));
DNN_BER_in_frame(Frame, 1) = DNN_numErrs_bit(Frame, 1) / length(DNN_Data_Received);

end

% BER calculation

DNN_BER_over_SNR(SNR + 1, 1) = sum(DNN_numErrs_bit, 1) / Num_of_bits_DNN;
DNN_SER_over_SNR(SNR + 1, 1) = sum(DNN_numErrs_sym, 1) / Num_of_QPSK_symbols_DNN;

end
