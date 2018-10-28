clear all
%load data
load('mnist.mat');

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
TrainY = y(1:TrainSize,:);
CV_Y = y(TrainSize+1:end,:);
testY = testY';

%due to memory limit, shrink train size
X = TrainX(1:10000,:);
y = TrainY(1:10000,:);

%construct neural network modal
NetworkLayers = 3;
Input_Neurons = 784;
Hiddden_Neurons = 15;
Output_Neurons = 10;

%plot data
sel = randperm(size(X,1));
sel = sel(1:100);
Display_Data(X(sel, :));
%Theta1 size: 15 * 784+1
%Theta2 size: 10 * 15+1

%randomly initialize theta
Init_Theta1 = Initialize_Theta(Input_Neurons,Hiddden_Neurons);
Init_Theta2 = Initialize_Theta(Hiddden_Neurons,Output_Neurons);
Init_Theta = [Init_Theta1(:);Init_Theta2(:)];
%Cost function regularized & grandient regularized
lambda = 0;
[J,Grad] = CostGradFunc(X,y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);

pause;

%gradient check
GradientCheck(lambda);
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

%Build Learning Curve to see next step
[error_train, error_val] = LearningCurve(TrainX(1:100,:),TrainY(1:100,:),CV_X(1:100,:),CV_Y(1:100,:),lambda,Input_Neurons,Hiddden_Neurons,Output_Neurons,Init_Theta);
plot(1:100, error_train, 1:100, error_val);
title('Learning curve for neural network')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 100 0 150])

%CV set for select lambda
