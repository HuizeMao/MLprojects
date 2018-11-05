function II = UserInputProcess(fileName)
  II = fileName;
  [p3, p4] = size(II); % 30  30 
  q1 = 28;  % size of the crop box
  i3_start = round((p3-q1)/2);  % 1 % or round instead of floor; using neither gives warning4
  i3_stop = p3 - i3_start;    % 
  
  i4_start = floor((p4-q1)/2);  % 1
  i4_stop = p4 - i4_start;   % 29
  while true
    if i3_start == 0 && i4_start == 0
       II = II;
       break
    end
    if i3_start == 1 && i4_start == 1
      II = II(i3_start + 1:i3_stop, i4_start + 1:i4_stop,:);
      break
    elseif i4_start == 1
      II = II(i3_start :i3_stop, i4_start + 1:i4_stop,:);
      break
    elseif i3_start == 1 
      II = II(i3_start + 1:i3_stop, i4_start:i4_stop,:);
      break
    else 
      II = II(i3_start:i3_stop, i4_start:i4_stop,:);
      break
    end
  endwhile
  %figure ,imshow(II); 