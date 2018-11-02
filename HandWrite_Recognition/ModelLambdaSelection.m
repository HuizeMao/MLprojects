function [lambda,hidden_layer_neurals] = ModelLambdaSelection(X,y,X_CV,y_CV,Input_Neurons,Output_Neurons)
  lambdaSelection = [1.3;1.33;1.36;1.4;2;10;5];
  HiddenNeural = [124];
  %costfunc = @(p,h,l) CostGradFunc(X,y,p,Input_Neurons,h,Output_Neurons,l);
  for i = 1 : length(lambdaSelection)
    for j = 1 : length(HiddenNeural)
      hiddenNeural = HiddenNeural(j,1);
      lambda = lambdaSelection(i);
      Init_Theta1 = Initialize_Theta(Input_Neurons,hiddenNeural);
      Init_Theta2 = Initialize_Theta(hiddenNeural,Output_Neurons);
      Init_Theta = [Init_Theta1(:);Init_Theta2(:)];
      theta = TrainNeurals(X,y,lambda,Init_Theta,Input_Neurons,hiddenNeural,Output_Neurons);
      J(i,j) = CostGradFunc(X_CV,y_CV,theta,Input_Neurons,hiddenNeural,Output_Neurons,lambda);
    end
  end
  J


%J = 2.3 -- 0.43
