function convoluted = cconv(A, B)
    outputSize = max(numel(A), numel(B));

    convoluted = zeros(1, outputSize);
    startIdx = floor((size(B, 2) - 1) / 2); % Starting index for convolution
    endIdx = size(A, 2) + floor((size(B, 2) - 1) / 2); % Ending index for convolution

    for x_A = 1:size(A, 2)
        for x_B = 1:size(B, 2)
            convIndex = x_A + x_B - startIdx;
            if convIndex >= 1 && convIndex <= outputSize
                convoluted(1, convIndex) = convoluted(1, convIndex) + A(1, x_A) * B(1, x_B);
            end
        end
    end

    convoluted = convoluted(1:size(A, 2));

end
