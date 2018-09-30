function [x] = SigmoidTranspose(X)
  x = X.*(1-X);
