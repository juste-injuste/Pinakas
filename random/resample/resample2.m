function resampled = resample2(data, L)
    data = reshape(data, 1, []);
    N	= numel(data);
	o	= 3 * L;
	l	= 2 * o + 1;
    s = (numel(data)-2)*L+1
    e = numel(data)*L-L;
    data = [2*data(1)-flip(data(2:end-1)), data, 2*data(end)-flip(data(2:end-1))];
    
    upsampled = zeros(1, numel(data)*L - L);
    upsampled((0:numel(data)-1)*L+1) = data;
    
	filter = sinc(((0:l-1)-o)/L) .* blackman(l)';
    
	resampled = conv(upsampled, filter)(s+o:s+o+e);
    
    figure, hold on
    plot(resampled)
end
