function [lambda,hidden_layer_neurals] = ModelLambdaSelection(X,y,X_CV,y_CV,Input_Neurons,Output_Neurons)
  lambdaSelection = [1.82];
  HiddenNeural = [100;200;300;400;500;600];
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
end

%J = 2.3 -- 0.43
%200
%2; 2.03; 3.06; 2.09; 2.12; 2.15; 2.18; 2.21; 2.24;2.27;2.3

%lambdaSelection = [0.01;0.05;0.5;1.3;1.4;2;5;10;15;20];
%HiddenNeural = [75;100;124;150;175;200];
%     2 J = 0.87183
%     1.82 156 J = 0.70292


%  lambdaSelection = [1.4;1.43;1.46;1.49;1.52;1.55;1.58;1.61;1.64;1.67;1.7;1.73;1.76;1.79;1.82;1.85;1.88;1.92;1.95;1.98;2];
%  HiddenNeural = [150;153;156;159;162;165;168];
