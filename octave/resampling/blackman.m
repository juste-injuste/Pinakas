function window = blackman(N)
    window = zeros(1, N);
    for i = 0:N-1
        window(i +1) = 0.42 - 0.5*cos(2*pi*i/(N-1)) + 0.08*cos(4*pi*i/(N - 1));    
    endfor
endfunction