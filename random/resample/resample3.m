function output = resample3(data, L)
    N        = numel(data);
	offset   = floor(L * 2.4)
	length   = 2*offset + 1;
    cut      = floor(N * 0.9)
    N_e      = (3*N - 2*cut) * L;
    extended = zeros(1, N_e);

    n = 0;
    for i = 0:(N-1)-cut
        extended(n+1) = 2*data(1) - data(N-i-cut+1);
        n = n + L;
    end
    
    start = n + 1 + 1
    for i = 0:N-1
        extended(n+1) = data(i + 1);
        n = n + L;
    end
    stop = n - L + 1 + 1
    
    for i = 0:(N-1)-cut
        extended(n+1) = 2*data(end) - data(N-2-i+1);
        n = n + L;
    end
    
	filter = sinc(((0:length-1)-offset)/L) .* blackman(length)';
    
    
    resampled = zeros(1, N_e);

    for x_A = 1:N_e
        for x_B = 1:length
            idx = x_A + x_B - offset;
            if idx >= 1 && idx <= N_e
                resampled(1, idx) += extended(x_A) * filter(x_B);
            end
        end
    end
    output = resampled(start:stop);
    numel(output)
    stop-start
    figure, hold on
    plot(resampled, 'b')
    plot(extended, 'k')
    plot([start start], [-2 2], 'r')
    plot([stop stop], [-2 2], 'r')
end
