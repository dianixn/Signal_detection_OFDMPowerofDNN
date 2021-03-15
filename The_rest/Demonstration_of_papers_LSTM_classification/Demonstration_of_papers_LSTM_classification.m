% Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems
% SNR represents Es/N0, Es/N0 = Eb/N0 * log2(M)
% For SNR, the power of pilot is ignored when calculating the varience of
% noise and the pilots suffer from the noise on equal level with data
% Power is balanced since IFFT/FFT is applied
% InsertDCNull is true and the effect caused by DCNull is considered in the
% SNR_OFDM, which is adjusted to be SNR + 10 * log(Num_of_subcarriers_used
% / Num_of_FFT)
% h_channel is different for each frame and stored in H, or read from h_set

SNR_Range = 5:25;
%h_set = [];
BER_over_SNR = zeros(length(SNR_Range), 1);
SER_over_SNR = zeros(length(SNR_Range), 1);
LSTM_BER_over_SNR = zeros(length(SNR_Range), 1);
LSTM_SER_over_SNR = zeros(length(SNR_Range), 1);

% Import Deep Neuron Network
load('Trained_LSTM_64.mat');

for SNR = SNR_Range

Baseband_bandwidth = 20e6;

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

Num_of_QPSK_symbols = Num_of_subcarriers * Num_of_symbols * Num_of_frame;
Num_of_bits = Num_of_QPSK_symbols * k;

Num_of_QPSK_symbols_LSTM = 2 * Num_of_symbols * Num_of_frame;
Num_of_bits_LSTM = Num_of_QPSK_symbols_LSTM * k;

numErrs_sym = zeros(Num_of_frame, 1);
SER_in_frame = zeros(Num_of_frame, 1);
numErrs_bit = zeros(Num_of_frame, 1);
BER_in_frame = zeros(Num_of_frame, 1);

LSTM_numErrs_sym = zeros(Num_of_frame, 1);
LSTM_SER_in_frame = zeros(Num_of_frame, 1);
LSTM_numErrs_bit = zeros(Num_of_frame, 1);
LSTM_BER_in_frame = zeros(Num_of_frame, 1);

Rayleigh_h = zeros(Num_of_frame * Frame_size * length_of_symbol, 1);
Multipath_h = zeros(Num_of_frame * Frame_size * length_of_symbol, 1);

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

data_in_IFFT = zeros(Num_of_FFT - 1, Frame_size);

data_in_IFFT(:, Pilot_location(:)) = Pilot_value;
data_in_IFFT(:, data_location(:)) = QPSK_signal;

data_in_IFFT = [zeros(1, Frame_size); data_in_IFFT];

% OFDM Transmitter
Transmitted_signal = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);



% Channel

% AWGN Channel
%Signal_power = sum(power(abs(Transmitted_signal), 2));
%Pilot_power = power(abs(Pilot_value), 2) * Num_of_Pilot_per_Frame;

Data_power = sum(power(abs(QPSK_symbol), 2)) / (length(QPSK_symbol));

SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
SNR_OFDM_HEX = 10 ^ (SNR_OFDM / 10);
Noise_power = Data_power / SNR_OFDM_HEX;

Nvariance = sqrt(Noise_power/2); % QPSK has two paths of signal
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation
AWGN_Signal = Transmitted_signal + n;

%channel = comm.AWGNChannel('NoiseMethod','Variance','VarianceSource','Input port');
%powerDB = 10*log10(var(Transmitted_signal));
%noiseVar = 10.^(0.1*(powerDB - SNR));
%Signal_1 = channel(Transmitted_signal,noiseVar);
%n_1 = Signal_1 - Transmitted_signal;

% Rayleigh Channel
Rayleigh_h_channel_OFDM_symbol = (1 / sqrt(2)) * randn(length_of_symbol, 1) + (1 / sqrt(2)) * 1j * randn(length_of_symbol, 1); % Rayleigh channel coff
Rayleigh_h_channel = repmat(Rayleigh_h_channel_OFDM_symbol, Frame_size, 1);

Rayleigh_Fading_Signal = Rayleigh_h_channel .* Transmitted_signal + n;
Rayleigh_h(((Frame - 1) * Frame_size * length_of_symbol) + 1 : Frame * Frame_size * length_of_symbol, 1) = Rayleigh_h_channel;

Channel_signal_when_h_is_known = Rayleigh_Fading_Signal ./ Rayleigh_h_channel;

% Multipath Rayleigh Fading Channel
Multipath_Fading_h_channel_OFDM_symbol = ((randn + 1j * randn) / sqrt(2)) * ones(length_of_symbol, 1);
Multipath_Fading_h_channel = repmat(Multipath_Fading_h_channel_OFDM_symbol, Frame_size, 1);

Multipath_Fading_Signal = Multipath_Fading_h_channel .* Transmitted_signal + n;
Multipath_h(((Frame - 1) * Frame_size * length_of_symbol) + 1 : Frame * Frame_size * length_of_symbol, 1) = Multipath_Fading_h_channel;

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
[Unrecovered_signal, Unrecovered_signal_when_h_is_known] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);

% Channel estimation

% Origin Received Signal
%Received_data = Unrecovered_signal(2:end, data_location(:));

% Perfect knowledge on Channel
Received_data_Perfect_knowledge = Unrecovered_signal_when_h_is_known(2:end, data_location(:));

% Zero-forcing
Received_pilot = Unrecovered_signal(:, Pilot_location(:));
H_LS = Received_pilot ./ Pilot_value;

Received_Signal_ZF = Unrecovered_signal(:, data_location(:)) ./ H_LS;
Received_data_ZF = Received_Signal_ZF(2:end, :);

% MMSE
% L = length_of_CP; change it to 0 and have a test, then compare it with L and ZF
%H_MMSE_L_0 = MMSE_Uniform_PDP(0, Num_of_FFT, 0, SNR, true, H_LS, H_LS); % Single tap channel
%H_MMSE_L_length_of_CP = MMSE_Uniform_PDP(length_of_CP, Num_of_FFT, 0, SNR, true, H_LS, H_LS);

H_MMSE_h = MMSE_Channel_Tap_Block_Pilot_Demo_1(Received_pilot, Pilot_value, Num_of_FFT, Frame_size, SNR, Multitap_h');
H_MMSE = H_MMSE_h;
Received_Signal_MMSE = Unrecovered_signal ./ H_MMSE;
Received_data_MMSE = Received_Signal_MMSE(2:end, data_location(:));

% Deep learning
% Detect 2 QPSK symbols only
[LSTM_feature_signal, ~, ~] = Extract_Feature_OFDM(Unrecovered_signal, dataSym(1:2), M, QPSK_signal(1:8));
Received_data_LSTM = classify(LSTM_Trained, LSTM_feature_signal);
Received_data_LSTM = grp2idx(Received_data_LSTM) - 1;

LSTM_Received_symbol = flip(rem(floor(Received_data_LSTM * M .^ (1 - length(dataSym(1:2)) : 0)), M), 2)';

Received_data = Received_data_MMSE;
%Received_data_ZF Received_data_MMSE Received_data_DNN
%scatterplot(Received_data(:, 1))

% Checking
Symbol_noise = Received_data - QPSK_signal;
Symbol_noise = reshape(Symbol_noise, [], 1);
Noise_power_Symbol = sum(power(abs(Symbol_noise), 2), 1)/ size(Symbol_noise, 1);
SNR_HEX_Symbol = 2 / Noise_power_Symbol;
SNR_Symbol = 10 * log10(SNR_HEX_Symbol);

% QPSK demodulator
dataSym_Rx = QPSK_Demodulator(Received_data);

dataSym_Received = de2bi(dataSym_Rx, 2);
Data_Received = reshape(dataSym_Received, [], 1);

LSTM_Received_data = de2bi(LSTM_Received_symbol, 2);
LSTM_Data_Received = reshape(LSTM_Received_data, [], 1);

% BER calculation in each frame
numErrs_sym(Frame, 1) = sum(sum(round(dataSym) ~= round(dataSym_Rx)));
SER_in_frame(Frame, 1) = numErrs_sym(Frame, 1) / length(dataSym);

numErrs_bit(Frame, 1) = sum(sum(round(Data) ~= round(Data_Received)));
BER_in_frame(Frame, 1) = numErrs_bit(Frame, 1) / length(Data);

% DNN BER calculation in each frame
LSTM_numErrs_sym(Frame, 1) = sum(sum(round(dataSym(1:2)) ~= round(LSTM_Received_symbol)));
LSTM_SER_in_frame(Frame, 1) = LSTM_numErrs_sym(Frame, 1) / length(LSTM_Received_symbol);

LSTM_numErrs_bit(Frame, 1) = sum(sum(round(reshape(de2bi(dataSym(1:2), 2),[],1)) ~= round(LSTM_Data_Received)));
LSTM_BER_in_frame(Frame, 1) = LSTM_numErrs_bit(Frame, 1) / length(LSTM_Data_Received);

end

% BER calculation
BER = sum(numErrs_bit, 1) / Num_of_bits;
SER = sum(numErrs_sym, 1) / Num_of_QPSK_symbols;

BER_over_SNR(SNR + 1, 1) = BER;
SER_over_SNR(SNR + 1, 1) = SER;

LSTM_BER_over_SNR(SNR + 1, 1) = sum(LSTM_numErrs_bit, 1) / Num_of_bits_LSTM;
LSTM_SER_over_SNR(SNR + 1, 1) = sum(LSTM_numErrs_sym, 1) / Num_of_QPSK_symbols_LSTM;

end
