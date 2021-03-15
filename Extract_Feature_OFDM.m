% 2020.06.29
% Extract the features from OFDM symbols
% The feature is composed of the real part and imaginary part of data

function [feature, label, label_regression] = Extract_Feature_OFDM(data, Data_symbol, Base, Data_label_regression)

feature = zeros(2 * numel(data), 1);

data2feature = reshape(data, [], 1);

Real_part = real(data2feature);
Imaginary_part = imag(data2feature);

label_regression = zeros(2 * numel(Data_label_regression), 1);

data2feature_regression = reshape(Data_label_regression, [], 1);

Real_part_regression = real(data2feature_regression);
Imaginary_part_regression = imag(data2feature_regression);

% Convert datasym to class
%label = full(data_symbol);
%[label, ~] = Onehot_generator(data_symbol + 1); % QPSK datasym is in the range of [0,3]
label = sum(power(Base, [0 : length(Data_symbol) - 1]) .* Data_symbol'); % One label for each combination

feature(1:2:end, 1) = Real_part;
feature(2:2:end, 1) = Imaginary_part;

label_regression(1:2:end, 1) = Real_part_regression;
label_regression(2:2:end, 1) = Imaginary_part_regression;

end
