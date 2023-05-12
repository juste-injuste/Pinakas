clc, clear, close all, pkg load signal


    close all
    N = 10;
    L = 100;
    x = linspace(0, 1, N);
    y = x*100;
    y = x.^2 + sin(x*5)/5 + rand(size(x))/5;
    y = y * 2 - max(2*y)/2;
    y2 = resample3(y, L);
    
    figure, hold on
    plot(x, y)
    plot(linspace(0, 1, numel(y2)), y2)
