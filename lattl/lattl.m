function [A, E, Phi, Psi, W, b] = lattl(Xs, Ys, A, Phi, Xt, Yt, E, Psi, W, b, params)
% Perform latent transductive transfer learning
%   [A, E, Phi, Psi, W, b] = lattl(Xs, Ys, Xt, Yt, Phi, Psi, W, b, params)
%
% Input
%   Xs:     n x d matrix of source instances (n=ex, d=feats)
%   Ys:     n x l vector of source labels
%   Phi:    d x r matrix of source latent mappings, initial value
%   Xt:     n x d matrix of target instances (n=ex, d=feats)
%   Yt:     n x l matrix of target labels or zeros when none
%   Psi:    d x r matrix of target latent mappings, initial value
%   W, b:   r x 1 weight vector w, bias b, initial value
%   params: set of nuisance parameters to control algorithm behavior:
%       - costs: C1, C3 (rec err); C2, C4 (class error)
%       - beta: regularization for representation learning
%       - r: latent space dimensionality
%       - epsilon: tolerance for convergence
%       - max_iter: max number of iterations
%
% Output
%   A:    n x r matrix of source latent space representations
%   E:    n x r matrix of target latent space representations
%   Phi:  d x r matrix of source latent mappings
%   Psi:  d x r matrix of target latent mappings
%   w, b: r x 1 weight vector w, bias b

% WARNING: original LATTL code was written for the multilabel setting, but
% recent modifications may assume single label setting.

% Indices of labeled, unlabeled target examples
labeled   = find(sum(Yt~=0,2) > 0);
unlabeled = find(sum(Yt~=0,2) == 0);

% We used to compute latent representations from using the mappings passed
% to the function, but now we'll accept A, E as inputs.
% QUESTION: revisit this question: Phi/Psi are basis functions (x = Phi * a) so
% is it appropriate to use them to go the other direction (a = Phi' * x)?
% ANSWER: yes, if they are orthogonal, which is the case when they are computed
% using PCA. So actually, we shouldn't be doing this in the general case
% A = Phi'*Xs';
% E = Psi'*Xt';

% remember W_old
W_old = W;

% no unlabeled target data? no point!
assert(length(unlabeled) > 0)

Cs = repmat(params.C2, size(Ys));
% adjust costs for class imbalance
Cs(Ys==1) = Cs(Ys==1) * params.cw(1);
Cs(Ys~=1) = Cs(Ys~=1) * params.cw(2);

Ct = repmat(params.C5, size(Yt));
Ct(unlabeled) = params.C4;
% adjust costs to class imbalance
Ct(Yt==1) = Ct(Yt==1) * params.cw(1) / sum(params.cw);
Ct(Yt~=1 & Yt~=0) = Ct(Yt~=1 & Yt~=0) * params.cw(2) / sum(params.cw);

% cram together labeled source/target labels, costs for TSVM training
Ylab = Ys;
Clab = Cs;
if sum(labeled) > 0
    Ylab = [ Ylab; Yt(labeled,:) ];
    Clab = [ Clab; Ct(labeled,:) ];
end

for i = 1:params.max_iter
    fprintf('Iteration %d\t', i)

    % Fix A, E; learn mappings, classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% BASIS FUNCTION LEARNING STEP (Problems 3, 4)
    % TODO: Use L1 norm to get sparsity
    fprintf('Basis...')
    tic;
    cst = cputime;
    % WARNING: stanford code takes matrices in d x n shape!    
    Phi = l2ls_learn_basis_dual(Xs', A', numel(Phi), Phi); % Problem 3
    Psi = l2ls_learn_basis_dual(Xt', E', numel(Psi), Psi); % Problem 4
    fprintf('%.5gct %.5gwt ', cputime-cst, toc);
    
    %% TSVM LEARNING STEP (Problem 5)
    fprintf('Classifiers...');
    tic;
    cst = cputime;
    % this for loop is an artifact of the multilabel version of TTL,
    % iterates over label dimensions
    for ii=1:size(Ys, 2)
        Xlab = A;
        if sum(labeled) > 0
            Xlab = [ Xlab; E(:, labeled) ];
        end
        [W(:, ii), b(:, ii)] = learnTsvm(W(:, ii), b(:, ii), Xlab, Ylab(:, ii), E(unlabeled, :), Clab(:, ii), params.C4);
%         [W(:, ii), b(:, ii)] = tsvmLearn(W(:, ii), b(:, ii), [ A'; E(:, labeled)'], [ Ys(:, ii); Yt(labeled, ii) ], E(:, unlabeled)', [ repmat(params.C2, size(Ys, 1), 1); repmat(params.C4, size(labeled, 1), 1) ], params.C3);
    end
    fprintf('%.5gct %.5gwt ', cputime-cst, toc);
    
    % Fix mappings, classifier; learng A, E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% SOURCE REPRESENTATION LEARNING STEP (Problems 6)
    fprintf('S Vectors...');
    cst = cputime;
    % this for loop iterates over samples
    for ii=1:size(Xs, 1)
        % TODO: modify aLearn to utilize different weights across classes
        [ A(ii, :), ~, ~,exitflag ] = learnLatentRep(W', b', Phi, Ys(ii, :)', Xs(ii, :)', params.C1, mean(Cs(ii, :))); % Problem 6
%         [ A(:, ii), ~, ~,~ ] = learnLatentRep(W', b', Phi, Ys(ii, :)', Xs(ii, :)', params.C1, mean(Cs(ii, :))); % Problem 6
        if exitflag ~= 1
            fprintf('exitflag for A: %d\n', exitflag)
        end
    end
    fprintf('%.5gct %.5gwt ', cputime-cst, toc);

    %% TARGET REPRESENTATION LEARNING STEP (Problems 7)
    fprintf('T Vectors...');
    Ythat = sign(get_svm_decision_value(W, b, E));
    Ythat(Ythat==0) = 1;
    Ythat(labeled, :) = Ylab(labeled, :);
    cst = cputime;
    % this for loop iterates over samples
    for ii=1:size(Xt, 1)
        % TODO: modify aLearn to utilize different weights across classes
        [ E(ii, :), ~, ~,exitflag ] = learnLatentRep(W', b', Psi, Ythat(ii, :)', Xt(ii, :)', params.C3, mean(Ct(ii, :))); % Problem 7
%         [ E(:, ii), ~, ~,~ ] = learnLatentRep(W', b', Psi, Ythat(ii, :)', Xt(ii, :)', params.C3, mean(Ct(ii, :))); % Problem 7
        if exitflag ~= 1
            fprintf('exitflag for E: %d\n', exitflag)
        end
    end
    fprintf('%.5gct %.5gwt ', cputime-cst, toc);
    fprintf('...DONE!\n')
    
    %% CHECK FOR CONVERGENCE
    if norm(W-W_old, 'fro')<params.epsilon
        fprintf('\n The difference is: %f\n', norm(W-W_old, 'fro'));
        break;
    end
end

%A = A';
%E = E';
%Phi = Phi';
%Psi = Psi';
