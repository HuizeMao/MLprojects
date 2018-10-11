%load data
load('mnist.mat');

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

%due to memory limit, shrink train size
X_part = TrainX(1:10000,:);
y_part = TrainY(1:10000,:);

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
Init_Theta = Initialize_Theta(Input_Neurons,Hiddden_Neurons,Output_Neurons);

%Cost function regularized & grandient regularized
lambda = 1;
[J,Grad] = CostGradFunc(X_part,y_part,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
%gradient check
GradientCheck(lambda);

%%Train Neural Network
fprintf('\nTraining Neural Network... \n')

theta = TrainNeurals(X_part,y_part,lambda,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons);

%resize Theta
num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
Theta1 = reshape(Init_Theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
Theta2 =  reshape(Init_Theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);

%predict the handwrite recognition and give accuracy
Prediction = predict(Theta1,Theta2,X);
accuracy = mean(double(Prediction == y)) * 100;
fprintf('\nLearning Accuracy: %f\n', accuracy)

%Build Learning Curve to see next step
[error_train, error_val] = LearningCurve(X_part(1:100,:),y_part(1:100,:),CV_X(1:100,:),CV_Y(1:100,:),lambda,Input_Neurons,Hiddden_Neurons,Output_Neurons,Init_Theta);
plot(1:100, error_train, 1:100, error_val);
title('Learning curve for neural network')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 100 0 150])

%CV set for select lambda
