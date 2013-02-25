function [w b] = newLearn(wInit, bInit, lsample, label, usample, parameters)

w = wInit;
b = bInit;
% disp(objective( label', lsample', usample', w, b, parameters))

for i = 1:10
    [alpha, alpha_b] = Update_Alpha(parameters.C3, usample, 0, w, b);
%     [ w2, ~, ~, b2 ] = wbLearn( parameters.C2, parameters.C3, alpha, alpha_b, label, lsample, usample, lsample, 0 );
    [ w, ~, ~, b ] = wbLearnGD( parameters.C2, parameters.C3, alpha, alpha_b, label, lsample, usample, lsample, 0 );
    disp(objective( label', lsample', usample', w, b, parameters))
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [alpha, alpha_b] = Update_Alpha(C, e, s, w, b)
    [ne,de] = size(e);
    % Mine
    a = sum([e, ones(ne, 1)].*repmat(( (e*w+b) < s), 1, de+1), 1) + ...
        sum(-[e, ones(ne, 1)].*repmat(( -(e*w+b) < s), 1, de+1), 1);
    a = a'*C;
    alpha = a(1:end-1);
    alpha_b = a(end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ w, epl, epu, b, fval, exitflag ] = wbLearnGD( C1, C2, a, ab, t, xl, xu, xa, wbc )
delta = 1e-5;
w = rand(size(xl, 2), 1);
b = 0.1;

Max_iter = 200;
obj = zeros(Max_iter, 1);
for i = 1:Max_iter
    [obj(i) Gw Gb] = findG( w, b, C1, C2, a, ab, t, xl, xu);
    w = w - delta*Gw;
    b = b - delta*Gb;
end
plot(obj)
epl = 0;
epu = 0;
fval = obj(end);
exitflag = 0;

end
function [obj Gw Gb] = findG( w, b, C1, C2, a, ab, yl, xl, xu)
obj = 0.5*norm(w)^2 + 0.5*b^2 - a'*w - ab*b;
obj = obj + C1*sum( (yl.*(xl*w+b)).*(yl.*(xl*w+b) > 0 ) );
obj = obj + C2*sum( abs(xu*w+b) );

Gb = -ab + b;
Gb = Gb + C1*sum( yl.*(yl.*(xl*w+b) > 0 ) );
Gb = Gb + C2*sum( (xu*w+b > 0) - (xu*w+b < 0) );

Gw = w - a;
Gw = Gw + C1*sum( xl .* repmat(yl .* (yl.*(xl*w+b) > 0 ), 1, length(w)) , 1)';
Gw = Gw + C2*sum( xu.* repmat((xu*w+b > 0) - (xu*w+b < 0), 1, length(w)) , 1)';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ w, epl, epu, b, fval, exitflag ] = wbLearn( C1, C2, a, ab, t, xl, xu, xa, wbc )

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
    
    opts = optimset('Algorithm','interior-point-convex','Display','off');
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
    e(1 : dx,:) = (sum(xu) / Nu - sum(xa) / Na)';
    
    H = zeros(dx + N + 1);
    H(1 : dx,1 : dx) = eye(dx);
    
    f = zeros(dx + N + 1,1);
    f(1 : dx, : ) = a;
    f((dx + 1) : (dx + Nl), : ) = C1;
    f((dx + Nl + 1) : (dx + Nl + Nu), : ) = C2;  %%% C2 -> C2/2 ?
    f(dx + N + 1, : ) = ab;
    
    A = zeros(2 * N,dx + N + 1);
    A(1 : N, 1 : dx) = D;
    A(1 : N, (dx + 1) : (dx + N)) = eye(N);
    A((N + 1) : (2 * N), (dx + 1) : (dx + N)) = eye(N);
    A(1 : N, dx + N + 1) = y;
    A = -A;
    
    u = zeros(2 * N,1);
    u(1 : N, : ) = -1;
    
    opts = optimset('Algorithm','interior-point-convex','Display','off');
    if wbc == 1
        [params,fval,exitflag] = quadprog(H,f,A,u,e',0,[],[],[],opts);
    else
        [params,fval,exitflag] = quadprog(H,f,A,u,[],[],[],[],[],opts);
    end
    w = params(1 : dx, : );
    epl = params((dx + 1) : (dx + Nl), : );
    epu = params((dx + 1) : (dx + Nu), : );
    b = params((dx + N + 1), : );
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = objective( label_set, A, E, W, b, parameters)
% First term
out = 0.5*( norm(W, 'fro')^2+ norm(b)^2); 
% Third Terms
xi = 1 - label_set .* (W'*A + repmat(b', 1, size(A, 2)));
out = out + parameters.C2*sum(sum((xi.*(xi>0))));
temp =  1-abs( W'*E + repmat(b', 1, size(E, 2)) );
out = out + parameters.C3*sum(sum( temp.*(temp > 0) ));

end