function [Theta] = TrainNeurals(X,y,lambda,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons)
  options = optimset('MaxIter', 1000);
  costFunction = @(p) CostGradFunc(X,y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
  [Theta] = fmincg(costFunction, Init_Theta, options);
