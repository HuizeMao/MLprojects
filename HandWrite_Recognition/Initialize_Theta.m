function Theta = Initialize_Theta(In_layer,Out_layer)
  epsilon = sqrt(6) / sqrt(In_layer + Out_layer);
  Theta = rand(Out_layer, In_layer+1) * 2 * epsilon - epsilon;
end
