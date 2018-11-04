function II = UserInputProcess(fileName)
  II = fileName
  [p3, p4] = size(II);
  q1 = 28;  % size of the crop box
  i3_start = floor((p3-q1)/2); % or round instead of floor; using neither gives warning
  i3_stop = i3_start + q1;

  i4_start = floor((p4-q1)/2);
  i4_stop = i4_start + q1;

  II = II(i3_start:i3_stop, i4_start:i4_stop, :);
  figure ,imshow(II);
