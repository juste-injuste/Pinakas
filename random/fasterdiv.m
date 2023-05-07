function x = fasterdiv(A, B)
    [M, N] = size(B);

    Q = rand(M, N);
    R = rand(N, N);
    x = rand(N, 1);

    for i = 1:N

        R(i, i) = 0;
        for j = 1:M
            R(i, i) = R(i, i) + B(j, i) * B(j, i);
        endfor
        R(i, i) = sqrt(R(i, i));
        for j = 1:M
            Q(j, i) = B(j, i) / R(i, i);
        endfor

        for k = i:N            
            R(i, i) = 0;
            for j = 1:M
                R(i, i) = R(i, i) + Q(j, i) * Q(j, i);
            endfor
            
            for j = 1:M
                Q(j, i) = Q(j, i) -  Q(j, i) * R(i, i);
            endfor
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
        x(i, 1) = substitution / R(i, i);
    endfor
endfunction



















