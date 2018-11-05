image(reshape(onesX(1,:),28,28))


trainX(401:600,:) = twosX(1:200,:);
CV_X(301:450,:) = twosX(201:350,:);
testX(381:570,:) = twosX(351:540,:);

%'UserInput0.jpg'

save MNIST_TenSaperate.mat zerosX onesX twosX threesX foursX fivesX sixesX sevensX eightsX ninesX
save InputData.mat CV_X CV_Y testX testY trainX trainY
trainX = [zerosX(1:200,:);onesX(1:200,:);twosX(1:200,:);threesX(1:200,:);...
  foursX(1:200,:);fivesX(1:200,:);sixesX(1:200,:);sevensX(1:200,:);...
   eightsX(1:200,:);ninesX(1:200,:)];

trainY(1:200,1) = 0;
trainY(201:400,1) = 1;
trainY(401:600,1) = 2;
trainY(601:800,1) = 3;
trainY(801:1000,1) = 4;
trainY(1001:1200,1) = 5;
trainY(1201:1400,1) = 6;
trainY(1401:1600,1) = 7;
trainY(1601:1800,1) = 8;
trainY(1801:2000,1) = 9;

   
CV_X = [zerosX(201:350,:);onesX(201:350,:);twosX(201:350,:);threesX(201:350,:);...
  foursX(201:350,:);fivesX(201:350,:);sixesX(201:350,:);sevensX(201:350,:);...
   eightsX(201:350,:);ninesX(201:350,:);];
CV_Y(1:150,1) = 0;
CV_Y(151:300,1) = 1;
CV_Y(301:450,1) = 2;
CV_Y(451:600,1) = 3;
CV_Y(601:750,1) = 4;
CV_Y(751:900,1) = 5;
CV_Y(901:1050,1) = 6;
CV_Y(1051:1200,1) = 7;
CV_Y(1201:1350,1) = 8;
CV_Y(1351:1500,1) = 9;

testX = [zerosX(351:540,:);onesX(351:540,:);twosX(351:540,:);threesX(351:540,:);...
  foursX(351:540,:);fivesX(351:540,:);sixesX(351:540,:);sevensX(351:540,:);...
   eightsX(351:540,:);ninesX(351:540,:)];


testY(1:190,1) = 0;
testY(191:380,1) = 1;
testY(381:570,1) = 2;
testY(571:760,1) = 3;
testY(761:950,1) = 4;
testY(951:1140,1) = 5;
testY(1141:1330,1) = 6;
testY(1331:1520,1) = 7;
testY(1521:1710,1) = 8;
testY(1711:1900,1) = 9;




%5499
ex_zeros = trainY == 0  % result in a vector in which if the element = 0 is 1, otherwise 0
p_zeros = find(ex_zeros) % find the position of 0s 
for i = 1: length(p_zeros)  
  zerosX(i,:) = trainX(p_zeros(i),:);
endfor
CV_X0 = zerosX(2001:3500,:);
test_X0 = zerosX(3501:5400,:);
%6742
ex_ones = trainY == 1
p_ones = find(ex_ones)
for i = 1: length(p_ones)
  cur = trainX(p_ones(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,90);
  onesX(i,:) = cur(:)';
endfor
CV_X1 = onesX(2001:3500,:);
test_X1 = onesX(3501:5400,:);

%5958
ex_twos = trainY == 2
p_twos = find(ex_twos)
for i = 1: length(p_twos)
  cur = trainX(p_twos(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  twosX(i,:) = cur(:)';
endfor

%6131
ex_threes = trainY == 3
p_threes = find(ex_threes)
for i = 1: length(p_threes)
  cur = trainX(p_threes(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,90);
  threesX(i,:) = cur(:)';
endfor

%5842
ex_fours = trainY == 4
p_fours = find(ex_fours)
for i = 1: length(p_fours)
  cur = trainX(p_fours(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  foursX(i,:) = cur(:)';
endfor

%5918
ex_fives = trainY == 5  % result in a vector in which if the element = 0 is 1, otherwise 0
p_fives = find(ex_fives) % find the position of 0s 
for i = 1: length(p_fives)  
  cur = trainX(p_fives(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  fivesX(i,:) = cur(:)';
endfor

%5918
ex_sixes = trainY == 6  % result in a vector in which if the element = 0 is 1, otherwise 0
p_sixes = find(ex_sixes) % find the position of 0s 
for i = 1: length(p_sixes)  
  cur = trainX(p_sixes(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  sixesX(i,:) = cur(:)';
endfor

%6265
ex_sevens = trainY == 7  % result in a vector in which if the element = 0 is 1, otherwise 0
p_sevens = find(ex_sevens) % find the position of 0s 
for i = 1: length(p_sevens)  
  cur = trainX(p_sevens(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  sevensX(i,:) = cur(:)';
endfor

%5851
ex_eights = trainY == 8  % result in a vector in which if the element = 0 is 1, otherwise 0
p_eights = find(ex_eights) % find the position of 0s 
for i = 1: length(p_eights)  
  cur = trainX(p_eights(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,90);
  eightsX(i,:) = cur(:)';
endfor

%5949
ex_nines = trainY == 9  % result in a vector in which if the element = 0 is 1, otherwise 0
p_nines = find(ex_nines) % find the position of 0s 
for i = 1: length(p_nines)  
  cur = trainX(p_nines(i),:);
  cur = reshape(cur,28,28);
  cur = imrotate(cur,270);
  cur = fliplr(cur);
  ninesX(i,:) = cur(:)';
endfor

%% Merge input data 
trainX60000 = [zerosX; onesX; twosX; threesX; foursX; fivesX; sixesX; sevensX; eightsX; ninesX];
trainX60000 = trainX60000(1:60000,:);
trainY = zeros(60000)