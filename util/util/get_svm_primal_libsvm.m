function [W, b] = get_svm_primal_libsvm(model)

%% LIBSVM with one output
W = model.SVs' * model.sv_coef;
b = -model.rho;
if model.Label(1) == -1
    W = -W;
    b = -b;
end
