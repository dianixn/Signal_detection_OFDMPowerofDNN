function H_MMSE = MMSE_Uniform_PDP(L, N, Trms, SNR, uniform_PDP, H_LS, h)

% 2020.05.20
% OFDM Channel Estimation by Singular Value Decomposition
% normalizing r_kk to unity
% A uniform power-delay profile can be obtained by letting Trms large
% enough
% For the scenario of single delay path, set L = 0
% h and H_LS should have a same size

Rhh = zeros(size(h, 1), size(h, 1));
SNR_HEX = 10^(SNR / 10);

if uniform_PDP
    for m = 0 : size(h, 1) - 1
        for n = 0 : size(h, 1) - 1
            if L == 0
                r_mn = exp(- 2 * pi * 1j * L * (m - n) / N);
            else
                r_mn = (1 - exp(- 2 * pi * 1j * L * (m - n) / N)) / (2 * pi * 1j * L * (m - n) / N);
            end
            if m == n
                Rhh(m + 1,n + 1) = 1;
            else
                Rhh(m + 1,n + 1) = r_mn;
            end
        end
    end
else
    for m = 0 : size(h, 1) - 1
        for n = 0 : size(h, 1) - 1
            r_mn = (1 - exp(- L * ((1 / Trms) + (2 * pi * 1j * (m - n))/ N))) / Trms * (1 - e^(- L / Trms)) * ((1 / Trms) + 2 * pi * 1j * (m - n) / N);
            if m == n
                Rhh(m + 1,n + 1) = 1;
            else
                Rhh(m + 1,n + 1) = r_mn;
            end
        end
    end
end

H_MMSE = Rhh * pinv(Rhh + (eye(size(Rhh , 1)) / SNR_HEX)) * H_LS;