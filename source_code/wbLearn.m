function [ w, epl, epu, b, fval, exitflag ] = wbLearn( C1, C2, a, ab, t, xl, xu, xa, wbc, opt_algo, large_scale )

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

if (Na == 0)
    
    xu = [xu;xu];
    D = [xl;xu];
    D = repmat(y,1,dx) .* D;
    
    H = zeros(dx + N + 1);
    H(1 : dx,1 : dx) = eye(dx);
    
    f = zeros(dx + N + 1,1);
    f(1 : dx, : ) = a;
    f((dx + 1) : (dx + Nl), : ) = C1;
    f((dx + Nl + 1) : (dx + Nl + Nu), : ) = C2;
    f(dx + N + 1, : ) = ab;
    
    A = zeros(2 * N,dx + N + 1);
    A(1 : N, 1 : dx) = D;
    A(1 : N, (dx + 1) : (dx + N)) = eye(N);
    A((N + 1) : (2 * N), (dx + 1) : (dx + N)) = eye(N);
    A(1 : N, dx + N + 1) = y;
    A = -A;
    
    u = zeros(2 * N,1);
    u(1 : N, : ) = -1;
    
    opts = optimset('Algorithm',opt_algo,'Display','off','LargeScale',large_scale);
    [params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);
    
    w = params(1 : dx, : );
    epl = params((dx + 1) : (dx + Nl), : );
    epu = params((dx + 1) : (dx + Nu), : );
    b = params((dx + N + 1), : );
    
else

    xu = [xu;xu];
    D = [xl;xu];
    D = repmat(y,1,dx) .* D;
    
    e = zeros(dx + N + 1,1);
    e(1 : dx,:) = (sum(xu) / Nu- sum(xa) / Na)'; %  
    bala = 0;
    
    H = zeros(dx + N + 1);
    H(1 : dx,1 : dx) = eye(dx); %%       ?????
    
    f = zeros(dx + N + 1,1);
    f(1 : dx, : ) = a;
    f((dx + 1) : (dx + Nl), : ) = C1;
    f((dx + Nl + 1) : (dx + Nl + Nu), : ) = C2;
    f(dx + N + 1, : ) = ab;
    
    A = zeros(2 * N,dx + N + 1);
    A(1 : N, 1 : dx) = D;
    A(1 : N, (dx + 1) : (dx + N)) = eye(N);
    A((N + 1) : (2 * N), (dx + 1) : (dx + N)) = eye(N);
    A(1 : N, dx + N + 1) = y;
    A = -A;
    
    u = zeros(2 * N,1);
    u(1 : N, : ) = -1;
    
    opts = optimset('Algorithm',opt_algo,'Display','off','LargeScale',large_scale);
    if (wbc == true)
        [params,fval,exitflag] = quadprog(H,f,A,u,e',bala,[],[],[],opts);
    else
        [params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);
    end
    
    w = params(1 : dx, : );
    epl = params((dx + 1) : (dx + Nl), : );
    epu = params((dx + 1) : (dx + Nu), : );
    b = params((dx + N + 1), : );
    
%     
%     % Gurobi
%     clear opts
%     c = f';
%     objtype = 1;    % minimization
%     B = sparse(A);
% %     b = u;
%     lb = -inf;        % [] means 0 lower bound %%%%%%%% Change this
%     ub = [];        % [] means inf upper bound
%     contypes = '<'; % all inequalities
%     vtypes = [];      % [] means all variables are continuous
%     opts.QP.qrow = int32([0:(dx-1), (dx+N)]); % indices of (x0), (x0), (x1) in 0.5 (x0)*x0 - (x0)*x1 + (x1)*x1;
%     opts.QP.qcol = int32([0:(dx-1), (dx+N)]); % indices of (x0), (x1), (x1) in 0.5 x0*(x0) - x0*(x1) + x1*(x1);
%     opts.QP.qval = ones(1, dx+1);     % coefficients of 0.5 x0^2 - x0*x1 + x1^2
%     
%     opts.IterationLimit = 200;
%     opts.FeasibilityTol = 1e-6;
%     opts.IntFeasTol = 1e-5;
%     opts.OptimalityTol = 1e-6;
%     opts.DisplayInterval = 0;
%     opts.OutputFlag = 0;
%     opts.Display = 0;
%     
% %     tic
%     [params,fval,exitflag] = gurobi_mex(c,objtype,B,u,contypes,lb,ub,vtypes,opts);
% %     toc
%     w = params(1 : dx, : );
%     b = params((dx + N + 1), : );
%     
%     epl = params((dx + 1) : (dx + Nl), : );
%     epu = params((dx + 1) : (dx + Nu), : );
end

end

