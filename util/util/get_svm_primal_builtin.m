function [W, b] = get_svm_primal_builtin(svm_struct)

W = -svm_struct.SupportVectors' * svm_struct.Alpha;
b = -svm_struct.Bias;

end