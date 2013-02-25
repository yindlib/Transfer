function [ a, e, fval, exitflag ] = aLearn( W, b, Phi, y, z, C1, C3, opt_algo, large_scale )

%{

Inputs:
W = [w1, w2 ... wl]' is a "l x da" matrix.
b is a "l x 1" vector.
Phi is a "dz x da" matrix.
y is a "l x 1" vector.
z is a "dz x 1" vector.
C1 & C3 are scalars.

Outputs:
a is a "da x 1" vector.
e is a "l x 1" vector, representing the slacknesses.
fval is a scalar, containing the value of the goal function.
exitflag is a scalar, indicating the status of the optimization process.

%}

[l, da] = size(W);
d = da + l;

H = zeros(d);
H(1 : da, 1 : da) = 2 * C1 * (Phi' * Phi);

f = zeros(d, 1);
f(1 : da, : ) = -2 * C1 * Phi' * z;
f((da + 1) : d, : ) = C3;

A = zeros(2 * l, d);
A(1 : l, 1 : da) = repmat(y, 1, da) .* W;
A(1 : l, (da + 1) : d) = eye(l);
A((l + 1) : (2 * l), (da + 1) : d) = eye(l);
A = -A;

u = zeros(2 * l, 1);
u(1 : l, :) = 1 - y .* b;
u = -u;

opts = optimset('Algorithm','interior-point-convex','Display','off','LargeScale','off');
[params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);

a = params(1 : da, : );
e = params((da + 1) : d, : );
fval = fval + C1 * (z') * z;


% 
% clear opts
% c = f';
% objtype = 1;    % minimization
% B = sparse(A);
% %     b = u;
% lb = -inf;        % [] means 0 lower bound %%%%%%%% Change this
% ub = [];        % [] means inf upper bound
% contypes = '<'; % all inequalities
% vtypes = [];      % [] means all variables are continuous
% temp = (0:(d-1))'*ones(1,d);
% opts.QP.qrow = int32(temp(:)'); 
% temp = temp';
% opts.QP.qcol = int32(temp(:)'); 
% opts.QP.qval = H(:)';     
% 
% opts.IterationLimit = 200;
% opts.FeasibilityTol = 1e-6;
% opts.IntFeasTol = 1e-5;
% opts.OptimalityTol = 1e-8;
% opts.DisplayInterval = 0;
% opts.OutputFlag = 0;
% opts.Display = 0;
% 
% %     tic
% [params,fval,exitflag] = gurobi_mex(c,objtype,B,u,contypes,lb,ub,vtypes,opts);
% %     toc
% a = params(1 : da, : );
% e = params((da + 1) : d, : );
% fval = fval + C1 * (z') * z;

end

