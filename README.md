# Signal_detection_OFDMPowerofDNN
MATLAB demonstration for the paper 'Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems' @ MATLAB R2020b

Rayleigh channel deployed, for the winner2 channel (Data_Generation_WIN2.m shows WIN2 for SISO) install that toolbox but more time will be spent on channel realization, so not suggested, and commu.AWGN(fading signal, SNR) not transmitted signal, I remembered that I % commu.AWGN codes

MMSE_Channel_Tap_Block_Pilot_Demo_1.m from MIMO-OFDM wireless communication book

MMSE_Uniform_PDP.m the paper of OFDM Channel Estimation by Singular Value Decomposition

Did not upload python version, there's a python demonstration uploaded by the author

Demonstration_of_papers_DNN is main

DNN_Regression_Image_SER_Test for taining

Dont need use Test_DNN_regression and Train_DNN necessarily

The_rest includes LSTM, classification and use sequence layer as input layer, but not that relative. I just tried to see if I can implement LSTM and what effects would be if I changed to different layers

update on 2021.04.10

Hi, someone has a question on Loss of DNN_Classification_Trained.mat. The solution is that, you need to run DNN_Regression_Image_SER_Test.m to ontain an DNN_trained which is a trained NN, and save that as XXX.mat with commend save('XXX.mat', 'DNN_trained'). In the Demonstraion file, there's a load commend, then change it to load('XXX.mat'). It allows you to make changes to training options so you can try some changes to see if you can improve the performance.
