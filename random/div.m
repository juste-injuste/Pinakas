function x = div(A, B)
    [M, N] = size(B);

    Q = zeros(M, N);
    R = zeros(N, N);
    x = zeros(N, 1);

    for i = 1:N
        R(i, i) = 0;
        for j = 1:M
            R(i, i) = R(i, i) + B(j, i) * B(j, i);
        endfor
        inorm = 1/sqrt(R(i, i));
        if (isfinite(inorm))
            for j = 1:M
                Q(j, i) = B(j, i) * inorm;
            endfor
        endif

        for k = i:N
            R(i, k) = 0;
            projection = 0;
            for j = 1:M
                R(i, k) = R(i, k) + Q(j, i) * B(j, k);
            endfor
            for j = 1:M
                B(j, k) = B(j, k) - R(i, k) * Q(j, i);
            endfor
            if (k >= i)
                R(i, k) = R(i, k);
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