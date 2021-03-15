function H_MMSE_h = MMSE_Channel_Tap_Block_Pilot_Demo_1(Received_Pilot, Pilot_Value, Nfft, Frame_size, SNR, h)

% 2020.06.09

H_MMSE_h = zeros(Nfft, Frame_size);

SNR_HEX = 10^(SNR / 10);
Np = Nfft;
H_LS = Received_Pilot ./ Pilot_Value;
Nps = 1;

k = 0: length(h) - 1;
hh = h * h';
tmp = h .* conj(h) .* k;
r = sum(tmp) / hh;
r2 = tmp * k .'/hh;

tau_rms = sqrt(r2 - r^2);
df = 1/Nfft;
j2pi_tau_df = 1j * 2 * pi * tau_rms * df;
K1 = repmat([0 : Nfft - 1].', 1, Np);
K2 = repmat([0 : Np - 1], Nfft, 1);
rf = 1./(1 + j2pi_tau_df * (K1 - K2 * Nps));
K3 = repmat([0 : Np - 1].', 1, Np);
K4 = repmat([0 : Np - 1], Np, 1);
rf2 = 1./(1 + j2pi_tau_df * Nps * (K3 - K4));
Rhp = rf;
Rpp = rf2 + (eye(length(H_LS)) / SNR_HEX);
H_MMSE = Rhp * pinv(Rpp) * H_LS;

for i_MMSE = 1 : Frame_size
    H_MMSE_h(:, i_MMSE) = H_MMSE;
end