while true
  str_input = input('Enter the name of the image with (. format) and put it within a single quotation mark: ');
%my1.jpg
  user_input = imread(str_input);
  %user_input = UserInputProcess(user_input)
  user_input = double(user_input(:)');
  prediction = predict(Theta1,Theta2,user_input);
  fprintf('\n')
  prediction
  fprintf('\n')
endwhile

%0 - ok
%1 - ok
%2 - ok
%3 - ok
%4 - ok
%5 - ok
%6 - ok
%7 - ok
%8 - ok
%9 - 3
