function [J] = CostGradFunctest(X,y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda)
  num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
  Theta1 = reshape(Init_Theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
  Theta2 =  reshape(Init_Theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);
  m = size(X,1);
  X = [ones(m,1),X];
  eye_matrix = eye(10);
  y_matrix = eye_matrix(y+1,:);
  a1 = X;
  z1 = Theta1 * a1';
  a2 = sigmoid(z1);
  one = ones(1,(size(X,1)));
  a2 = [one; a2];
  z2 = Theta2 * a2;
  hypo = sigmoid(z2);

  hypo = hypo(:,1:10000);
  y = y_matrix(1:10000,:);

  J = J + (1/m * (-y_matrix * log(hypo) - (1-y_matrix) * log(1 - hypo)));
  J = trace(J);
  regurized_term = (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
  J = J + regurized_term;
