function x = fastdiv(A, B)
    [M, N] = size(B);

    Q = zeros(M, N);
    R = zeros(N, N);
    x = zeros(N, 1);

    for j = 1:N
        sum_of_squares = 0;
        for i = 1:M
            sum_of_squares = sum_of_squares + B(i, j) * B(i, j);
        endfor
        inorm = 1/sqrt(sum_of_squares);
        if (isfinite(inorm))
            for i = 1:M
                Q(i, j) = B(i, j) * inorm;
            endfor
        endif

        for k = j:N
            projection = 0;
            for i = 1:M
                projection = projection + Q(i, j) * B(i, k);
            endfor
            for i = 1:M
                B(i, k) = B(i, k) - projection * Q(i, j);
            endfor
            if (k >= j)
                R(j, k) = projection;
            endif
        endfor
    endfor
    
    for i = N:-1:1
        substitution = 0;
        for j = 1:M
            substitution = substitution + Q(j, i) * A(j, 1);
        endfor
        for j = N:-1:i+1
            substitution = substitution - R(i, j) * x(j, 1);
        endfor
        x(i) = substitution / R(i, i);
    endfor
endfunction