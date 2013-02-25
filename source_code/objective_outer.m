function [obj_val] = objective_outer(W, b, Phi, beta, C1, C2, C3, z, a, y, e, s)

obj_val = 0;
[m, ~] = size(z);
[~, l] = size(W);
[n, ~] = size(e);
ee = [e;e];

% first term.
obj_val = obj_val + sum(sum(W .* W)) / 2;

% second term.
temp = z - a * Phi';
obj_val = obj_val + C1 * (sum(sum(temp .* temp)) + beta * m * sum(sum(Phi .* Phi)));

% third term.
temp = H(y .* (a * W + repmat(b,m,1)), 1);
obj_val = obj_val + C2 * sum(sum(temp));

% fourth term.
temp = H([ones(n,l); - ones(n,l)] .* (ee * W + repmat(b,2 * n,1)), 1);
obj_val = obj_val + C3 * sum(sum(temp));

% fifth term.
temp = H([ones(n,l); - ones(n,l)] .* (ee * W + repmat(b,2 * n,1)), s);
obj_val = obj_val - C3 * sum(sum(temp));

end