clc, clear, close all


x = [1:5]';

%y = 0.01*x.^4 + 0.1*x.^3 + 2*x.^2 - x.^1 + 3*x.^0;
y = x.^2 - x + 3;
w = [x.^2 x.^1 x.^0];
%w = [x.^4 x.^3 x.^2 x.^1 x.^0];

fastdiv(y, w)
div(y, w)