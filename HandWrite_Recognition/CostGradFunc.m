function [J] = CostGradFunc(X,Y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons)
  num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
  theta1 = reshape(Init_Theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
  theta2 =  reshape(Init_Theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);
  %Theta1 size: 15 * 784+1
  %Theta2 size: 10 * 15+1
  %X size: 1 * 784
  n = size(X,2);
  a1 = [1;X(:)];
  %a1 size: 785 * 1
  z2 = theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  %a2 size: 16 * 1
  z3 = theta2 * a2;
  hypo = sigmoid(z3);
  %hypo size: 10 * 1
  eye_matrix = eye(Output_Neurons);
  y_matrx = eye_matrix(y,:)
  J = (1/m) * ((-y_matrx * log(hypo)) - (1-y_matrix) * log(1-hypo));

end
