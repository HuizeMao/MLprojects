function Theta = Initialize_Theta(nl, nl2, nl3)
  theta1 = rand(nl2,(nl+1)); %size: 15*784
  theta2 = rand(nl3,(nl2+1)); %size: 10 * 15
  epsilon = 1.5;
  theta1 = theta1 * (2 * epsilon) - epsilon;
  theta2 = theta2 * (2 * epsilon) - epsilon;
  Theta = [theta1(:);theta2(:)];
end
