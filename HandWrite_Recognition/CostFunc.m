function [J] = CostFunc(X,Y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda)
  num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
  theta1 = reshape(Init_Theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
  theta2 =  reshape(Init_Theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);
  %Theta1 size: 15 * 784+1
  %Theta2 size: 10 * 15+1
  %X size: 48000 * 784
  m = size(X,1);
  n = size(X,2);
  a1 = [ones(m,1),X];
  %a1 size: 48000 * 785
  z2 = theta1 * a1';
  %z2 size: 15 * 48000
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2));a2];
  %a2 size: 16 * 48000
  z3 = theta2 * a2;
  hypo = sigmoid(z3);
  %hypo size: 10 * 48000
  eye_matrix = eye(Output_Neurons);
  y_matrix = eye_matrix(Y+1,:);
  J = (1/m) * ((-y_matrix * log(hypo)) - (1-y_matrix) * log(1-hypo));
  J = trace(J);
end
