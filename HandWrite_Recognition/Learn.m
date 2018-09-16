load("mnist.mat");
X = trainX;
y = trainY;
m = size(X,1);
n = size(X,2);

NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 15;
Output_Neurons = 10;

Init_Theta = Initialize_Theta(Input_Neurons,Hiddden_Neurons,Output_Neurons);
