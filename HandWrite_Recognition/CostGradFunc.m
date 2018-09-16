function [cost, grad] = CostGradFunc(X,Y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons)
  theta1 = reshape(Init_Theta(1:11760),Hiddden_Neurons,Input_Neurons);
  theta2 =  reshape(Init_Theta(11761:end),Hiddden_Neurons,Output_Neurons);
  a1 = [1;X];
  z2 = theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = theta2 * a2;
  a3 = sigmoid(z3);
end
