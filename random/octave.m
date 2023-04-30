clc, clear, close all




x = [0:9]';
y = x.^2 - x + 3;
x_work = [x.^2, x.^1, x.^0];
[Q, R] = qr(x_work)
Q*R
x_work\y;