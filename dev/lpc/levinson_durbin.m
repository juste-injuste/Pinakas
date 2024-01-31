% DOESNT WORK, BUT IS CLOSE TO !

function A = levinson_durbin(V)
    % Initialize the first two coefficients
    P = -V(2) / V(1);
    K = V(1) + P * V(2);
    A = [1, P];

    for m = 3:length(V)
        % Calculate reflection coefficient
        P = -A(end:-1:1) * V(2:m).' / K;

        % Update K
        K = K * (1 - P^2);

        % Update the LPC coefficients
        A = [A, P];
        for i = 2:(m-1)/2 + 1
            A(i) = A(i) + P * A(m-i+1);
            A(m-i+1) = A(m-i+1) + P * A(i);
        end
    end
end
