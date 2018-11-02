function [error_train,error_val] = LearningCurve(X,y,X_val,y_val,lambda,Input_Neurons,Hiddden_Neurons,Output_Neurons,Init_Theta)
  m = size(X,1);
  for i = 1:m
    X_matrix = X(1:i,:);
    y_vector = y(1:i);
    theta = TrainNeurals(X_matrix,y_vector,lambda, Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons);
    error_train(i) = CostGradFunc(X_matrix,y_vector,theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
    error_val(i) = CostGradFunc(X_val,y_val,theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
end
