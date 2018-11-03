while true
  str_input = input('Enter the name of the image with (. format) and put it within a single quotation mark: ');
%my1.jpg
  user_input = imread(str_input);
  %user_input = UserInputProcess(user_input, cropPercentage=0, rotStep=0)
  user_input = double(user_input(:)');
  prediction = predict(Theta1,Theta2,user_input);
  fprintf('\n')
  prediction
  fprintf('\n')
endwhile


%1 -
%2 -
%3 -
%4 -
%5 -
%6 -
%7 -
%8 -
%9 -
