clc, clear, close all


A = rand(80000, 16);
b = rand(80000, 1);

tic;
A\b;
toc;