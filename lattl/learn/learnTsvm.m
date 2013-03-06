function [w, b, succ] = learnTsvm(wInit, bInit, lsample, label, usample, Cl, Cu)
% Learning for the LATTL transductive SVM
%   [w, b, succ] = newLearnModified(wInit, bInit, lsample, label, usample, Cl, Cu)
%
% Input
%   wInit:   r x 1 weight vector, initial value
%   bInit:   bias
%   lsample: n x r matrix of labeled instances (n=ex, r=feats)
%   label:   n x l vector of labels
%   usample: m x r matrix of unlabeled instances (n=ex, r=feats)
%   Cl:      n x 1 vector of costs for labeled samples
%   Cu:      n x 1 vector of costs for unlabeled samples
%
% Output
%   w:     r x 1 weight vector, initial value
%   b:     bias
%   succ:  binary flag indicating success or failure of optimization

w = wInit;
b = bInit;

fval_prev = inf;
succ = 0;
% mini = 10;
mini = 5;
maxi = 20;
i = 1;
fprintf('learnTsvm')
while i < maxi
    fprintf('...iter %d:', i)
    [alpha, alpha_b] = learnTsvmUpdateAlpha(Cu, usample, 0, w, b);
    wold = w;
    bold = b;
    [ w, ~, ~, b, fval, exitflag ] = learnTsvmWb(Cl, Cu, alpha, alpha_b, label, lsample, usample, lsample, 0, 'interior-point-convex', 'off');
    if exitflag < 0
        w = wold;
        b = bold;
    else
        succ = 1;
    end
    if i >= mini & succ
        fprintf('(iter threshold) ')
        break
    end
%     if abs(fval - fval_prev) <= 1E-7
%         fprintf('(converged early) ')
%         break
%     end
    if i >= maxi
        fprintf('(iter max, no soln?) ')
        break
    end
    i = i + 1;
    fval_prev = fval;
end
