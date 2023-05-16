clc, clear, close all, pkg load signal
    f = @(x) x.^2 + sin(x*5)/3 - 1;% + rand(size(x))/5;
    N = 10;
    L = 10;
    x = linspace(0, 1, N);
    y = f(x);

    %{
      %r = rand(1, N)/5;
      y += [0.016748	0.1511 0.19835	0.033717	0.033579	0.051496	0.058398	0.10786	0.065196	0.061863	0.04363	0.10378	0.16575	0.05308	0.055851	0.015496	0.07512	0.18187	0.030945	0.048376	0.19328	0.070647	0.099789	0.10904	0.02914	0.1071	0.17111	0.17751	0.040571	0.059446	0.007785	0.13023	0.068409	0.18275	0.15642	0.14127	0.19554	0.033098	0.014157	0.17072	0.15477	0.10364	0.13605	0.19833	0.088625	0.19727	0.14842	0.060845	0.044022	0.16741	0.12924	0.11696	0.05974	0.10885	0.11788	0.089201	0.14689	0.035279	0.074471	0.087861	0.167	0.1512	0.11929	0.19999	0.026522	0.19822	0.1726	0.09603	0.14319	0.10574	0.042014	0.13876	0.16961	0.021735	0.13893	0.1937	0.032036	0.094687	0.11584	0.16091	0.1575	0.08644	0.0058914	0.15119	0.13863	0.0092059	0.10256	0.14353	0.060306	0.1686	0.17302	0.13475	0.052569 0.14519 0.027774	0.024348	0.062468	0.18529	0.13628	0.053443];
    %}

    xOpt = linspace(0, 1, N*L-L+1);
    yOpt = f(xOpt);

    y2 = pinakas_resample(y, L);
    %y3 = resample(y, L, 1);
    

    % {
    figure, hold on
    plot(x, y, "xk") %, "linestyle", '-'
    plot(linspace(x(1), x(end), numel(y2)), y2, 'b')
    %plot(linspace(x(1), x(end), numel(y3)), y3, 'r')
    plot(xOpt, yOpt, 'm')
    %}