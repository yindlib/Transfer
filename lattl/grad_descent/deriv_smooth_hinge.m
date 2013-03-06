function val = deriv_smooth_hinge(z)

val = zeros(size(z));
val(z <= 0) = -ones(size(z(z <= 0)));
val(z < 1) = z(z < 1) - 1;

