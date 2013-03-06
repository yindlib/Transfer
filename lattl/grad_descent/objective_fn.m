function val = objective_fn(y, A, w, b)

val = sum(smooth_hinge(y .* (A * w + b))) / size(y,1);
