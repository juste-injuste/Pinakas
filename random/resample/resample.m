function resampled = resample(data, L)
    N	= numel(data)*L;
	o	= 3 * L;
	l	= 2 * o + 1;

    upsampled = upsample(data, L);

	impulse = sinc(((0:l-1)-o)/L);
    
    window = blackman(l);
    
	resampled = zeros(1, N);
	for i = 0:N-1
		for n = 0:l-1
            if (i+n-o) < N-1
                idx = abs(i+n-o);
            else
                idx = 2*N-1 - (i+n-o);
            end
			resampled(i+1) = resampled(i+1) + upsampled(idx+1) * impulse(n+1) * window(n+1);
		end
	end
end
