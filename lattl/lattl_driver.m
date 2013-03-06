%% Nuisance paramters
params.C1       = 0.1; %0.1;     % C1: cost, Err_recon(A)   (source)
params.C2       = 5; %10;      % C2: cost, Err_svm(A,y)   (source, labeled)
params.C3       = 0.5; %0.3;     % C3: cost, Err_recon(E)   (target)
params.C4       = 10; %10;      % C4: cost, Err_tsvm(E,y*) (target, unlabeled)
params.C5       = 100;      % C5: cost, Err_tsvm(E,y)  (target, labeled)
params.beta     = 0.00001; % beta: regularization
params.r        = 2;       % r: latent space dimension
params.epsilon  = 0.01;    % epsilon: tolerance for parameters
params.max_iter = 3;       % maximum no. of iterations

% A convenience for libsvm: libsvm assigns numbers to labels (class 1,
% class 2, etc.) based on the order it finds them in training data. By
% sorting them in descending order, we ensure that y=1 is #1, y=-1 is #2,
% etc.
[ys, idx ] = sort(ys, 'descend');
Xs = Xs(idx,:);

% idx = balance_dataset(ys);
% Xs = Xs(idx,:);
% ys = ys(idx);

% LATTL actually accepts labels in the target domain, but we're not using
% them so create a vector of all zeros
labels = zeros(size(Xt,1),1);

% Class weights: this is one strategy for handling class imbalance
params.cw = [ min(numel(ys) / sum(ys == 1), 20), min(numel(ys) / sum(ys ~= 1), 20) ]

% TODO: fiddle with nuisance parameters!
C2 = 10; %0.1; %logspace(-2, 2, 8);
C3 = 0.3; %100; %logspace(-2, 2, 8);
for i=1:length(C2)
    % Initialize latent space (mapping, representation) with PCA
    [Phi_pca, A_pca] = pca_basis(Xs, params.r);
    [Psi_pca, E_pca] = pca_basis(Xt, params.r);
%     Phi_pca = eye(2);
%     Psi_pca = eye(2);
%     A_pca = Xs;
%     E_pca = Xt;
    
    % Initialize [w,b] with libsvm SVM, get baseline classification results
    svmparams = sprintf('-t 0 -h 0 -c %f -w1 %f -w2 %f', ...
                        params.C2, params.cw(1), params.cw(2))

    model = svmtrain2(ys, A_pca, svmparams);
    [yhat, ~, ydv] = svmpredict(yt, E_pca, model);
    
    % Evaluate performance
    perf = eval_fave_metrics(yt, yhat, ydv);
    perf.recon = norm(Xt - E_pca * Psi_pca', 'fro') / size(E_pca, 1);
%     R2 = (1/300)*(sqrt(trace(E_pca'*E_pca)));
%     fprintf('PCA+SVM: %s    R2=%1.5g\n', s, R2)
    fprintf('PCA+SVM: a=%1.5g p=%1.5g r=%1.5g f1=%1.5g mcc=%1.5g auroc=%1.5g auprc=%1.5g recon=%1.5g\n', ...
            perf.a, perf.p, perf.r, perf.f1, perf.mcc, perf.auroc, perf.auprc, perf.recon);
    
    % TTL needs the primal parameters
    [ Wsvm, bsvm] = get_svm_primal_libsvm(model);
    for j = 1:length(C3)
        [A, E, Phi, Psi, W, b] = lattl(Xs, ys, A_pca, Phi_pca, Xt, labels, E_pca, Psi_pca, Wsvm, bsvm, params);
        
        % Get decision values and evaluate model
        ydv = get_svm_decision_value(W, b, E);
        yhat = sign(ydv);
        yhat(yhat==0) = 1;
        perf = eval_fave_metrics(yt, yhat, ydv);
        perf.recon = norm(Xt - E * Psi', 'fro') / size(E, 1);
%         R2 = (1/300)*(sqrt(trace(E_pca'*E_pca)));
%         fprintf('LATTL: %s    R2=%1.5g\n', s, R2)
        fprintf('LATTL: a=%1.5g p=%1.5g r=%1.5g f1=%1.5g mcc=%1.5g auroc=%1.5g auprc=%1.5g recon=%1.5g\n', ...
                perf.a, perf.p, perf.r, perf.f1, perf.mcc, perf.auroc, perf.auprc, perf.recon);
    end
end