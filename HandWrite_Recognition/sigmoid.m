function g = sigmoid(theta,X)
  g = 1/(1 + exp(X * theta'))
end
