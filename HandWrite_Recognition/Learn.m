clear all
clc

%load data
load('mnist.mat');

%convert 0s to 10s for X and y(they suppose to be 10)
for i = 1 : length(trainY)
	if trainY(i) == 0
	trainY(i) = 10;
end
end
for i = 1 : length(testY)
	if testY(i) == 0
	testY(i) = 10;
end
end

% Define some useful variables
X = double(trainX); % X size: 60,000 *784
y = double(trainY); % y size: 60,000 * 1
y = y';
m = size(X,1); % number of training examples
n = size(X,2);% number of features

%set Cross Validation set
TrainSize = m * 0.8;
CVSize = m * 0.2;

TrainX = X(1:TrainSize,:);
CV_X = X(TrainSize+1:end,:);
TrainY = y(1:TrainSize,:);
CV_Y = y(TrainSize+1:end,:);
testX = double(testX);
testY = double(testY');

%due to memory limit, shrink train size
X = TrainX(1:1000,:);
y = TrainY(1:1000,:);

%construct neural network architecture
NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 100;
Output_Neurons = 10;

%plot part of input data
sel = randperm(size(X,1));
sel = sel(1:100);
Display_Data(X(sel, :));

%randomly initialize theta
Init_Theta1 = Initialize_Theta(Input_Neurons,Hiddden_Neurons); %Theta1 size: 15 * 784+1
Init_Theta2 = Initialize_Theta(Hiddden_Neurons,Output_Neurons); %Theta2 size: 10 * 15+1
Init_Theta = [Init_Theta1(:);Init_Theta2(:)];

%Use cost function to compute cost&grandient
lambda = 0.01; %first set lambda(penalty for weights to prevent overfitting)
[J,Grad] = CostGradFunc(X,y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
J
%gradient check(check if backpropagation is implemented correctly)
GradientCheck(lambda);
pause;
fprintf('\nloading errors for differ modal + lambda... \n')
ModelLambdaSelection(CV_X,CV_Y,Input_Neurons,Output_Neurons);
pause;

%%Train Neural Network
fprintf('\nTraining Neural Network... \n')
theta = TrainNeurals(X,y,lambda,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons);

%resize Theta
num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
Theta1 = reshape(theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
Theta2 =  reshape(theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);

%predict the handwrite recognition and give accuracy
Prediction = predict(Theta1,Theta2,X);
accuracy = mean(double(Prediction == y)) * 100;
fprintf('\nLearning Accuracy: %f\n', accuracy)
%Test accuracy
TestPrediction = predict(Theta1,Theta2,testX);
accuracyTest = mean(double(TestPrediction == testY)) * 100;
fprintf('\nTest Accuracy: %f\n', accuracyTest)

%Build Learning Curve to diagnose
[error_train, error_val] = LearningCurve(X(1:50,:),y(1:50,:),CV_X(1:50,:),CV_Y(1:50,:),lambda,Input_Neurons,Hiddden_Neurons,Output_Neurons,Init_Theta);
plot(1:50, error_train, 1:50, error_val);
title('Learning curve for neural network')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 50 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end
%CV set for select lambda
