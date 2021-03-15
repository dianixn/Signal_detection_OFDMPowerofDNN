figure;
semilogy([0:25],BER_over_SNR,'b');
hold on
semilogy([0:25],DNN_BER_over_SNR,'r');

legend('BER when MMSE are applied over Multitap tap channel', ...
    'BER when DNN are applied to classify over Multitap tap channel');

xlabel('SNR in dB');
ylabel('Bit Error Rate');
title('BER');
grid on;
hold off;