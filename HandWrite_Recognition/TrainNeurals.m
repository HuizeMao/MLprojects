function [Theta] = TrainNeurals(X,y,lambda,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons)
  options = optimset('MaxIter', 100);
  costFunction = @(p) CostGradFunc(X,y,p,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda);
  [Theta,cost] = fmincg(costFunction, Init_Theta, options);
