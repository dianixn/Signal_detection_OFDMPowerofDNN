% 2020.06.30
% One hot encoding for constellation symbols

function [label, label_char] = Onehot_generator(data_symbol)

one_hot_vec = full(ind2vec(data_symbol', max(data_symbol)));
label = reshape(one_hot_vec, [], 1);
label_char = num2str(label);
end