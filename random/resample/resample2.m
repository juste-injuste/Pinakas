function resampled = resample2(data, L)
    N	= numel(data);
	o	= 3 * L;
	l	= 2 * o + 1;
    s = (numel(data)-2)*L+1
    e = numel(data)*L-L;
    edata = [2*data(1)-flip(data(2:end-1)), data, 2*data(end)-flip(data(2:end-1))];
    tdata = zeros(3*numel(data)-2);
    size(edata)
    size(tdata)
    upsampled = zeros(1, numel(edata)*L - L);
    upsampled((0:numel(edata)-1)*L+1) = edata;
    
	filter = sinc(((0:l-1)-o)/L) .* blackman(l)';
    
	resampled = conv(upsampled, filter);
    idx = 1;
    for index = s+o:s+o+e
        output(idx) = resampled(index);
        idx = idx + 1;
    endfor
    figure, hold on
    plot(linspace(0, 1, numel(output)), output)
    plot(linspace(0, 1, numel(data)), data)
end
