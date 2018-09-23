load("mnist.mat");
 % X size: 60,000 *784
X = double(trainX);
y = double(trainY); % y size: 60,000 * 1
y = y';
m = size(X,1);
n = size(X,2);
%set Cross Validation set
TrainSize = m * 0.8;
CVSize = m * 0.2;

TrainX = X(1:TrainSize,:);
CV_X = X(TrainSize+1:end,:);
testX = testX;
TrainY = y(1:TrainSize,:);
CV_Y = y(TrainSize+1:end,:);
testY = testY';

NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 15;
Output_Neurons = 10;

%Theta1 size: 15 * 784+1
%Theta2 size: 10 * 15+1
Init_Theta = Initialize_Theta(Input_Neurons,Hiddden_Neurons,Output_Neurons);
%Cost function regularized
lambda = 0;
J = CostFunc(TrainX,TrainY,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
