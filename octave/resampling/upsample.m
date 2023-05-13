function upsampled = upsample(data, L)
    upsampled = zeros(1, numel(data)*L-L);
    upsampled((0:numel(data)-1)*L+1) = data;
end