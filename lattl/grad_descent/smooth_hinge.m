function val = smooth_hinge(z)

val = zeros(size(z));
val(z <= 0) = 0.5 - z(z <= 0);
val(z < 1) = 0.5 .* (1-z(z < 1)).^2;
 