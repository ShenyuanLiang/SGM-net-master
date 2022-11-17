function val = L2Dist(q1,q2)

[n,T] = size(q1);
val = sqrt(trapz(linspace(0,1,T),sum((abs(q1-q2)).^2)));

return;