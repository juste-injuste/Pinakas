clc, clear, close all


data = [1, 0.5, 0.3, 0.2];

LPC = lpc(data, 3)

LPC = levinson_durbin(data);

fprintf('\nLPC Coefficients:\n');
disp(LPC);
