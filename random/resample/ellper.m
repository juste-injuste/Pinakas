function perimeter = ellper(a, b, approximation)
    try
        switch(approximation)
            case 'Ramanujan'
                perimeter   = pi*(3*(a+b)-sqrt((3*a+b).*(a+3*b)));
            case 'Ramanujan2'
                h           = (a-b).^2./(a+b).^2;
                perimeter   = pi*(a+b).*(1+3*h./(10+sqrt(4-3*h)));
            case 'Asselin'
                A           = [3.95297, 7.55787, 7.42507, 2.64605, 0.158914];
                B           = [1, 1.75926, 0.66118, 0.0397297];
                r           = min(b./a, a./b);
                perimeter   = max(a, b).*polyval(A, r)./polyval(B, r);
            case 'Asselin2' % [4 3], 5 digits, 5000 ppz
                A           = [3.94893, 7.329265, 7.121278, 2.417196, 0.1340207];
                B           = [1, 1.696845, 0.6040559, 0.03350577];
                r           = min(b./a, a./b);
                perimeter   = max(a, b).*polyval(A, r)./polyval(B, r);
                
        end
    catch
        e           = sqrt(a.^2-b.^2)./a;
        for idx = 1:numel(a+b)
            perimeter(idx)  = real(integral(@(x) 4*a.*sqrt(1-e(idx).^2*sin(x).^2), 0, pi/2));
        end
    end
end
