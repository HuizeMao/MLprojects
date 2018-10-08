function [J, Grad] = CostGradFunc(X,Y,Init_Theta,Input_Neurons,Hiddden_Neurons,Output_Neurons,lambda)
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
  y_matrix = eye_matrix(Y+1,:); % size = 48000 * 10

  regularized_term = lambda/(2*m) * (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)));
  J = 1/m * ((-y_matrix * log(hypo)) - ((1-y_matrix) * log(1-hypo)));
  J = trace(J);
  J = J + regularized_term;


%Calculate Gradient
  OutPutGrad = hypo - y_matrix' ; %size = 10 * 48000
  Second_delta = (theta2' * OutPutGrad) .* SigmoidTranspose(a2); % size = 16 * 48000
  Second_delta = Second_delta(2:end,:);% size = 15 * 48000
  regularized_1 = (lambda/m) * theta1(:,2:end);
  regularized_2 = (lambda/m) * theta2(:,2:end);
  Theta1_grad = (1/m) * Second_delta * a1; % 15 * 785
  Theta2_grad = (1/m) * OutPutGrad * a2'; % 10 * 16
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end); + regularized_1;
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end); + regularized_2;
  Grad = [Theta1_grad(:);Theta2_grad(:)];
