function resampled = resample(data, L, keep, alpha)
  N      = numel(data);
  offset = floor(L * alpha);
  length = 2*offset + 1;
  first  = L*keep;
  last   = L*(keep + N-1);
  extended = zeros(1, (N + 2*keep) * L);
  
  k = 0;
  for i = 0:keep-1
    extended(k +1) = 2*data(0 +1) - data(keep-i +1);
    k += L;
  end
  for i = 0:N-1
    extended(k +1) = data(i +1);
    k += L;
  end
  for i = 0:keep-1
    extended(k +1) = 2*data(N-1 +1) - data(N-2-i +1);
    k += L;
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

  %{
  figure, hold on
  plot([first:last]+1, resampled, 'b')
  stem(extended, 'k', "Marker", "none")
  plot([first first]+1, [-2 2], 'r')
  plot([last last]+1, [-2 2], 'r')
  %}
end
