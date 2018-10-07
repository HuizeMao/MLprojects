function [J, Grad] = CostGradFuncDebug(X,Y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda)
  num_theta1 = Hiddden_Neurons * (Input_Neurons +1);
  theta1 = reshape(Init_Theta(1:num_theta1),Hiddden_Neurons,Input_Neurons+1);
  theta2 =  reshape(Init_Theta(num_theta1+1:end),Output_Neurons,Hiddden_Neurons+1);
  m = size(X,1);
  n = size(X,2);
  a1 = [ones(m,1),X];
  z2 = theta1 * a1';
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2));a2];
  z3 = theta2 * a2;
  hypo = sigmoid(z3);
  eye_matrix = eye(Output_Neurons);
  y_matrix = eye_matrix(Y,:);
 
  regularized_term = lambda/(2*m) * (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)));
  J = 1/m * ((-y_matrix * log(hypo)) - ((1-y_matrix) * log(1-hypo)));
  J = trace(J);
  J = J + regularized_term;


%Calculate Gradient
  OutPutGrad = y_matrix' - hypo; 
  Second_delta = (theta2' * OutPutGrad) .* SigmoidTranspose(a2);
  Second_delta = Second_delta(2:end,:);
  regularized_1 = (lambda/m) * theta1(:,2:end);
  regularized_2 = (lambda/m) * theta2(:,2:end);
  Theta1_grad = (1/m) * Second_delta * a1; 
  Theta2_grad = (1/m) * OutPutGrad * a2'; 
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end); + regularized_1;
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end); + regularized_2;
  Grad = [Theta1_grad(:);Theta2_grad(:)];
