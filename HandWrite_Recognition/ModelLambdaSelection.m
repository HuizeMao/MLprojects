function [lambda,hidden_layer_neurals] = ModelLambdaSelection(X,y,Input_Neurons,Output_Neurons)
  lambdaSelection = [0.01;0.04;0.07;0.1;0.13;0.16;0.19;0.23;0.25;0.28;0.31;0.34;0.37;0.4];
  HiddenNeural = [100;103;106;109;112;115;118;121;124;127;130];
  %costfunc = @(p,h,l) CostGradFunc(X,y,p,Input_Neurons,h,Output_Neurons,l);
  for i = 1 : length(lambdaSelection)
    for j = 1 : length(HiddenNeural)
      hiddenNeural = HiddenNeural(j,1);
      lambda = lambdaSelection(i);
      Init_Theta1 = Initialize_Theta(Input_Neurons,hiddenNeural);
      Init_Theta2 = Initialize_Theta(hiddenNeural,Output_Neurons);
      Init_Theta = [Init_Theta1(:);Init_Theta2(:)];
      theta = TrainNeurals(X,y,lambda,Init_Theta,Input_Neurons,hiddenNeural,Output_Neurons);
      J(i,j) = CostGradFunc(X,y,theta,Input_Neurons,hiddenNeural,Output_Neurons,lambda);
    end
  end
  J
