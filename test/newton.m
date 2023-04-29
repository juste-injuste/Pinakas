function root = newton(f, seed, varargin)
    tolerance       = 0.0001;
    max_iterations  = 50;
    bound           = 5;
    points_per_unit = 10/(2*bound*+1);
    
    for idx = 1:2:numel(varargin)
        switch(varargin{idx})
            case 'tolerance'
                tolerance       = varargin{idx+1};
            case 'max_iterations'
                max_iterations  = varargin{idx+1};
            case 'bound'
                bound           = varargin{idx+1};
                points_per_unit = 10/(2*bound*+1);
            case 'points_per_unit'
                points_per_unit = varargin{idx+1};
            otherwise
                error(['unknown option : ' varargin{idx}])
        end
    end

    % create initial seeds if none are specified
    switch(seed)
        case 'real'
            seed    = [-bound:bound];
        case 'complex'
            % create complex seed plane
            seed    = repmat([-bound:1/points_per_unit:bound],  points_per_unit*2*bound+1, 1)...
                    - repmat([-bound:1/points_per_unit:bound]', 1, points_per_unit*2*bound+1)*i;
            % remove atleast 21.5% of the seeds
            seed(abs(seed) > bound) = [];
    end

    % newton-raphson method for finding roots
    dfdx        = @(x) (f(x+0.0001)-f(x))/0.0001;
    root        = seed;
    itt         = zeros(numel(root));
    tolerance
    for idx = 1:numel(root)
        while abs(f(root(idx))) > tolerance && itt(idx) < max_iterations
            root(idx)
            f(root(idx))
            root(idx)       = root(idx) - f(root(idx))/dfdx(root(idx));
            itt(idx)        = itt(idx) + 1;
        end
    end
    
    % remove Inf/NaN/non-roots answers
    root(~isfinite(root)||abs(f(root)) > tolerance/10) = [];
    % remove duplicate roots and round to tolerance
    root = unique(round(root/tolerance)*tolerance);
    % output  column vector
    root = reshape(root, [], 1);
end