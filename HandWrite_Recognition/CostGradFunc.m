function [cost, grad] = CostGradFunc(x)
  cost = x +1
  grad = x ^2
end
