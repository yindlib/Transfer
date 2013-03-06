function [ w, epl, epu, b, fval, exitflag ] = learnTsvmWb(Cl, Cu, a, ab, t, xl, xu, xa, wbc, opt_algo, large_scale)

use_gurobi = 1;

% Inputs:
% C1 & C2 are scalars.
% a is a 'd x 1' vector for w.
% ab is a scalar for b.
% t is a 'Nl x 1' vector.
% xl is a 'Nl x d' matrix.
% xu is a 'Nu x d' matrix.
% xa is a 'Na x d' matrix.
% wbc is a scalar.

% Outputs:
% w is a 'd x 1' vector.
% epl is a 'Nl x 1' vector.
% epu is a 'Nu x 1' vector.
% b is a scalar.
% fval is a scalar.
% exitflag is a scalar.

[Nl,dx] = size(xl);
[Nu,~] = size(xu);
[Na,~] = size(xa);
Nu = 2 * Nu;
N = Nl + Nu;

y = zeros(N,1);
y(1 : Nl, : ) = t;
y((Nl + 1) : (Nl + Nu / 2), : ) = 1;
y((Nl + Nu / 2 + 1) : N, : ) = -1;

xu = [xu;xu];
D = [xl;xu];
D = repmat(y,1,dx) .* D;

H = zeros(dx + N + 1);
H(1 : dx,1 : dx) = eye(dx);

f = zeros(dx + N + 1,1);
f(1 : dx, : ) = a;
f((dx + 1) : (dx + Nl), : ) = Cl;
f((dx + Nl + 1) : (dx + Nl + Nu), : ) = Cu;
f(dx + N + 1, : ) = ab;

A = zeros(2 * N,dx + N + 1);
A(1 : N, 1 : dx) = D;
A(1 : N, (dx + 1) : (dx + N)) = eye(N);
A((N + 1) : (2 * N), (dx + 1) : (dx + N)) = eye(N);
A(1 : N, dx + N + 1) = y;
A = -A;

u = zeros(2 * N,1);
u(1 : N, : ) = -1;

if use_gurobi == 0
    if (Na == 0)
        opts = optimset('Algorithm',opt_algo,'Display','off','LargeScale',large_scale);
        [params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);
    else
        e = zeros(dx + N + 1,1);
        e(1 : dx,:) = (sum(xu) / Nu- sum(xa) / Na)';
        bala = 0;
    
        opts = optimset('Algorithm',opt_algo,'Display','off','LargeScale',large_scale);
        if wbc == 1
            [params,fval,exitflag] = quadprog(H,f,A,u,e',bala,[],[],[],opts);
        else
            [params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);
        end
    end
    
    w = params(1 : dx, : );
    epl = params((dx + 1) : (dx + Nl), : );
    epu = params((dx + 1) : (dx + Nu), : );
    b = params((dx + N + 1), : );

else
    clear oparams;
%     oparams.QP.qrow = int32([0:(dx-1), (dx+N)]); % indices of (x0), (x0), (x1) in 0.5 (x0)*x0 - (x0)*x1 + (x1)*x1;
%     oparams.QP.qcol = int32([0:(dx-1), (dx+N)]); % indices of (x0), (x1), (x1) in 0.5 x0*(x0) - x0*(x1) + x1*(x1);
%     oparams.QP.qval = ones(1, dx+1);     % coefficients of 0.5 x0^2 - x0*x1 + x1^2
    
%     oparams.IterationLimit = 500;
    oparams.TimeLimit      = 6000;
    oparams.FeasibilityTol = 1e-5;
    oparams.OptimalityTol  = 1e-5;
%     oparams.DisplayInterval = 0;
    oparams.OutputFlag     = 0;
    oparams.LogToConsole   = 0;

    %% GUROBI MEX
%     c = f';
%     objtype = 1;    % minimization
%     B = sparse(A);
% %     b = u;
%     lb = -inf;        % [] means 0 lower bound %%%%%%%% Change this
%     ub = [];        % [] means inf upper bound
%     contypes = '<'; % all inequalities
%     vtypes = [];      % [] means all variables are continuous
    
%     [params,fval,exitflag] = gurobi_mex(c,objtype,B,u,contypes,lb,ub,vtypes,oparams);
    %%
    
    %% GUROBI 5.1
    clear omodel;
    omodel.obj = f';  % linear objective vector
    omodel.modelsense = 'min';  % 'min' or 'max, 'min' is default value
    omodel.A = sparse(A);  % linear constraint matrix
    omodel.rhs = u;  % rhs of constraints...
    omodel.lb = -inf + zeros(size(A,2),1);  % lower bound  <- TODO: change this???
%     omodel.ub = ?;  % upper bound, inf is default
%     TODO: specify constraint types?
    omodel.sense = '<';  % type of constraint, <, =, or <
%     omodel.vtypes = ?;     % variable types, continuous is default

    oresult = gurobi(omodel, oparams);
    exitflag = oresult.status;
    if isfield(oresult, 'x') & ~strcmp(exitflag, 'LOADED') & ...
       ~strcmp(exitflag, 'INFEASIBLE') & ~strcmp(exitflag, 'INF_OR_UNBD') & ...
       ~strcmp(exitflag, 'UNBOUNDED') & ~strcmp(exitflag, 'CUTOFF') & ...
       ~strcmp(exitflag, 'ITERATION_LIMIT') & ~strcmp(exitflag, 'NODE_LIMIT') & ...
       ~strcmp(exitflag, 'TIME_LIMIT') & ~strcmp(exitflag, 'NUMERIC')
        fval = oresult.objval;
        params = oresult.x;
        w   = params(1 : dx, : );
        b   = params((dx + N + 1), : );
        epl = params((dx + 1) : (dx + Nl), : );
        epu = params((dx + 1) : (dx + Nu), : );
        fprintf('converged\n');
        exitflag = 1;
    else
        w = -inf;
        b = -inf;
        epl = -inf;
        epu = -inf;
        fprintf('%s\n', exitflag)
        exitflag = -1;
    end
end
