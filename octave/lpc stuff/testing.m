clc, clear, close all, pkg load signal
time   = linspace(0, 1, 100);
signal = time.^2;


signal(31) = 1.5;
signal(32) = 1.7;
signal(33) = 1.9;
signal(34) = 1.9;
signal(35) = 1.7;
signal(36) = 1.5;

lpc(signal, 5);