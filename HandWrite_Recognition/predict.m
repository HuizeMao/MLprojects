function p = predict(theta1,theta2,X)
  m = size(X,1);
  n = size(theta2,1);
  L2 = sigmoid([ones(m,1) X] * theta1');
  pred = sigmoid([ones(m,1) L2] * theta2'); % size = m * 10
  [dummy,p] = max(pred,[],2); % p is the position in rows(which is the prediction)
  p(p == 10) = 0;  % convert 10 to 0
