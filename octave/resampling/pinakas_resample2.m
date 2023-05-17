function resampled = pinakas_resample2(data, L, keep, alpha, tail)
    if nargin < 3
        keep = 2;
    endif
    if nargin < 4
        alpha = 3.5;
    endif
    if nargin < 5
        tail = false;
    endif
    
    N      = numel(data);
    offset = floor(L * alpha);
    length = 2*offset + 1;
    
    first = L*keep;
    last  = L*(N-1) + first;
    
    extended = zeros(1, N + 2*keep);
    k = 0;
    for i = 0:keep-1
        extended(k +1) = 2*data(0 +1) - data(keep-i +1);
        k += 1;
    end
    for i = 0:N-1
        extended(k +1) = data(i +1);
        k += 1;
    end
    for i = 0:keep-1
        extended(k +1) = 2*data(N-1 +1) - data(N-2-i +1);
        k += 1;
    end
    filter = sinc(((0:length-1)-offset)/L) .* blackman(length);
    resampled = zeros(1, last - first + 1);
    for x_A = 0:numel(extended)-1
        for x_B = 0:length-1
            idx = x_A*L + x_B - offset;
            if (first <= idx && idx <= last)
                resampled(idx-first +1) += extended(x_A +1) * filter(x_B +1);
            end
        end
    end

    % {
    figure, hold on
    plot([first:last]+1, resampled, 'b')
    stem([[0:numel(extended)-1]]*L+1, extended, 'k', "Marker", "x")
    plot([first first]+1, [-1.2 1.2], 'r')
    plot([last last]+1, [-1.2 1.2], 'r')
    %}
end
