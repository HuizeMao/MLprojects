load("mnist.mat");
X = trainX;
y = trainY;
m = size(X,1);
n = size(X,2);

NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 20;
Output_Neurons = 10;
