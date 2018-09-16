load("mnist.mat");
X = trainX; % X size: 60,000 *784
y = trainY;
m = size(X,1);
n = size(X,2);

NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 15;
Output_Neurons = 10;
%Theta1 size: 15 * 784
%Theta2 size: 10 * 15
Init_Theta = Initialize_Theta(Input_Neurons,Hiddden_Neurons,Output_Neurons);
%
