function convoluted = cconv(A, B)
    lenA = size(A, 2);
    lenB = size(B, 2);
    lenC = lenA + lenB - 1;
    start = floor((lenB - 1) / 2) + 1;
    stop = start + lenA - 1;
    convoluted = zeros(1, lenC);

    % compute cropped convolution
    for x_A = 1:lenA
        x_B_start = max(start - x_A, 1);
        x_B_end = min(stop - x_A + 1, lenB);
        for x_B = x_B_start:x_B_end
            convoluted(x_A) = convoluted(x_A) + A(x_A - x_B + start) * B(x_B);
        end
    end

    convoluted = convoluted(start:stop);
end