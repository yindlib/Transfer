function [gw, gb, gA] = grad_objective(y, A, w, b)

temp1 = y .* deriv_smooth_hinge(y .* (A * w + b));
temp2 = repmat(temp1, 1, size(A,2));
gw = sum(A.*temp2) / size(y,1);
gb = sum(temp1) / size(y,1);
gA = repmat(w', size(A,1), 1) .* temp2 / size(y,1);


