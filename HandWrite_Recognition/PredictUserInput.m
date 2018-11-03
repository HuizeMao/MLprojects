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

%0 - 0  0
%1 - 6  4  y
%2 - 2  2
%3 - 3  2  y
%4 - 4  7  
%5 - 2  3
%6 - 9  9
%7 - 4  2
%8 - 2  2
%9 - 4  4
