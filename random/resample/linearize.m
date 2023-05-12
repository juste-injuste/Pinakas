function [new_x, new_y] = linearize(xdata, ydata)
    n = numel(xdata);
    new_x = zeros(1, n);
    new_y = zeros(1, n);
    step = (xdata(n)-xdata(1))/(n-1);

    new_x(1)    = xdata(1);
	new_y(1)    = ydata(1);
    new_x(n)    = xdata(n);
    new_y(n)    = ydata(n);

    for i = 2:n-1
        new_x(i) = new_x(i-1) + step;

        x1 = xdata(i);
		y1 = ydata(i);
        x2 = xdata(i+1);
        y2 = ydata(i+1);

        new_y(i) = ((y1-y2)*new_x(i) + x1*y2 - x2*y1)/(x1-x2);
    end
end
