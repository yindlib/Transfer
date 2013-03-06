function [ objval, gw, gb, gA, gE ] = total_objective(y, A, Z, E, X, w, b)

objval = objective_fn(y, A, w, b);
objval = objval + objective_fn(-ones(size(E,1),1), E, w, b);
objval = objval + objective_fn(ones(size(E,1),1), E, w, b);
objval = objval + 0.5 * norm(w)^2;
objval = objval + norm(A - Z, 'fro')^2 / size(A,1);
objval = objval + norm(E - X, 'fro')^2 / size(E,1);

gw = w;
gb = 0;
gA = 2 * (A - Z) / size(A,1);
gE = 2 * (E - X) / size(E,1);

[gwt, gbt, gAt] = grad_objective(y, A, w, b);
gw = gw + gwt';
gb = gb + gbt;
gA = gA + gAt;

[gwt, gbt, gEt] = grad_objective(-ones(size(E,1),1), E, w, b);
gw = gw + gwt';
gb = gb + gbt;
gE = gE + gEt;

[gwt, gbt, gEt] = grad_objective(ones(size(E,1),1), E, w, b);
gw = gw + gwt';
gb = gb + gbt;
gE = gE + gEt;

