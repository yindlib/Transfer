clc;

y = randi(2, 50, 1) - 1;
y(y~=1) = -1;

A = rand(50, 10);
Z = rand(size(A));

E = rand(50, 10);
X = rand(size(E));

w = rand(10, 1);
b = rand(1, 1);

delta = 1E-8;

%obj1 = objective_fn(y, A, w, b);
%[ gw1, gb1, gA1 ] = grad_objective(y, A, w, b);
[ obj1, ~, ~, ~, ~ ] = total_objective(y, A, Z, E, X, w, b);
[ obj2, gw, gb, gA, gE ] = total_objective(y, A, Z, E, X, w, b + delta);

disp(gb)
disp((obj2 - obj1)/delta)

% obj1 = objective_fn(y, A, w, b);
% w(1) = w(1) + delta;
% obj2 = objective_fn(y, A, w, b);
% w(1) = w(1) - delta;

[ obj1, ~, ~, ~, ~ ] = total_objective(y, A, Z, E, X, w, b);
w(1) = w(1) + delta;
[ obj2, gw, gb, gA, gE ] = total_objective(y, A, Z, E, X, w, b);
w(1) = w(1) - delta;
disp(gw(1))
disp((obj2 - obj1)/delta)

% obj1 = objective_fn(y, A, w, b);
[ obj1, ~, ~, ~, gE ] = total_objective(y, A, Z, E, X, w, b);
% disp(gA(2,3))
E(2,3) = E(2,3) + delta;
[ obj2, gw, gb, gA, gE ] = total_objective(y, A, Z, E, X, w, b);
% obj2 = objective_fn(y, A, w, b);
disp(gE(2,3))
disp((obj2 - obj1)/delta)

