clc, clear, close all


a = [0 0.2 0.4 0.6 0.8   1 1.2 1.4 1.6 1.8   2 1.8 1.6 1.4 1.2   1 0.8 0.6 0.4 0.2   0];
b = [1 1 1];

t = conv(a, b)

figure, plot(t)