function resampled = resample3(data, L, keep, impulse_length_factor)
  % keep = 2
  % impulse_length_factor = 3.5
  N        = numel(data);
  offset   = floor(L * impulse_length_factor);
  length   = 2*offset + 1;

  drop  = N - keep;
  extended = zeros(1, (3*N - 2*drop) * L);
  n = 0;
  for i = 0:(N-1)-drop
    extended(n +1) = 2*data(0 +1) - data(N-i-drop +1);
    n += L;
  end
  first = n;
  for i = 0:N-1
    extended(n +1) = data(i +1);
    n += L;
  end
  last = n - L;
  for i = 0:(N-1)-drop
    extended(n +1) = 2*data(N-1 +1) - data(N-2-i +1);
    n += L;
  end

  filter = sinc(((0:length-1)-offset)/L) .* blackman(length)';
  resampled = zeros(1, last - first + 1);
  for x_A = 0:numel(extended)-1
    for x_B = 0:length-1
      idx = x_A + x_B - offset;
      if (first <= idx && idx <= last)
        resampled(idx-first +1) += extended(x_A +1) * filter(x_B +1);
      end
    end
  end

  % {
  figure, hold on
  plot([first:last]+1, resampled, 'b')
  stem(extended, 'k', "Marker", "none")
  plot([first first]+1, [-2 2], 'r')
  plot([last last]+1, [-2 2], 'r')
  %}
end
