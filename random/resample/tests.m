clc, clear, close all

f = @(x) ellper(x, 1, 'Asselin') #+ sin(10*x);
L = 10;
N = 20;

xdata   = linspace(0, 1, N)';
xdata   = sort(xdata + [rand(floor(numel(xdata)/2), 1); -rand(ceil(numel(xdata)/2), 1)]/(5*N));
ydata   = f(xdata) + rand(size(xdata))/100;
truex = linspace(0, 1, 100);
truey = f(truex);
[xt, yt] = linearize(xdata, ydata);

yflt = resample2(yt, L);
xflt = linspace(0, 1, numel(yflt));

wyflt = resample2(ydata, L);
wxflt = linspace(0, 1, numel(wyflt));

%{
figure, hold on
plot(xdata, ydata, 'ok')
plot(xt, yt, 'or')
plot(xflt, yflt, 'b')
plot(wxflt, wyflt, 'm')
plot(truex, truey, 'k')
%}
figure, hold on
plot(xdata, ydata, "x")
plot(xflt, yflt)
csvwrite("data.csv", [xdata, ydata])
save("data.mat", 'xdata', 'ydata')
