function [lambda,hidden_layer_neurals] = ModelLambdaSelection(X,y,Input_Neurons,Output_Neurons,Init_Theta)
  lambdaSelection = [0];
  hiddenNeural = [14;15];
  %costfunc = @(p,h,l) CostGradFunc(X,y,p,Input_Neurons,h,Output_Neurons,l);
  for i = length(lambdaSelection)
    for j = length(hiddenNeural)
      Init_Theta1 = Initialize_Theta(Input_Neurons,hiddenNeural(j));
      Init_Theta2 = Initialize_Theta(hiddenNeural(j),Output_Neurons);
      Init_Theta = [Init_Theta1(:);Init_Theta2(:)];
      theta = TrainNeurals(X,y,lambdaSelection(i),Init_Theta,Input_Neurons,hiddenNeural(j),Output_Neurons);
      J(i,j) = CostGradFunc(X,y,theta,Input_Neurons,hiddenNeural(j),Output_Neurons,lambdaSelection(i));
    end
  end
  J
