% QPSK Demodulator
% Average signal power is 2, 1 + 1j, 1 - 1j, - 1 + 1j, - 1 - 1j
% Gray coding

function dataSym = QPSK_Demodulator(Received_signal)

N = size(Received_signal, 1) * size(Received_signal, 2);

dataSym = zeros(N, 1);

dataSym(find((angle(Received_signal) > 0) .* (angle(Received_signal) < (pi * 1/2)))) = 0;
dataSym(find((angle(Received_signal) > (pi/2)) .* (angle(Received_signal) < pi))) = 1;
dataSym(find((angle(Received_signal) > (- pi)) .* (angle(Received_signal) < (- pi/2)))) = 3;
dataSym(find((angle(Received_signal) > (- pi/2)) .* (angle(Received_signal) < 0))) = 2;