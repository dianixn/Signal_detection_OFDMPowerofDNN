function Transmitted_Signal = OFDM_Transmitter(Data_in_IFFT, IFFT_Size, CP_Size)

% IFFT
data_in_CP = ifft(Data_in_IFFT);
data_in_CP = sqrt(IFFT_Size) * data_in_CP;

% CP
Signal_from_baseband = [data_in_CP(IFFT_Size - CP_Size + 1 : end, :); data_in_CP];

% P2S
Transmitted_Signal = reshape(Signal_from_baseband, [], 1);
