% QPSK Modulator
% Average signal power is 2, 1 + 1j, 1 - 1j, - 1 + 1j, - 1 - 1j
% Gray coding

function QPSK_symbol = QPSK_Modualtor(dataSym)

N = size(dataSym, 1) * size(dataSym, 2);

QPSK_symbol = zeros(N, 1);

QPSK_symbol(( dataSym == 0)) = 1 + 1j;
QPSK_symbol(( dataSym == 1)) = -1 + 1j;
QPSK_symbol(( dataSym == 3)) = -1 - 1j;
QPSK_symbol(( dataSym == 2)) = 1 - 1j;