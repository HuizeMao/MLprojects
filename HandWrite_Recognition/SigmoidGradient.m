function [x] = SigmoidTranspose(X)
  x = sigmoid(X) .* (1-sigmoid(X));
end