% pca analysis: m is the dimension of the subspace to project to
function [PCAMat, PCAVal] = pca(m, data)
% m: 1 x 1 ... dimensionality of the projection space
% data: n x d ... design matrix

mean_data = mean(data);
% make sure data is centralized
data = data - mean_data(ones(1,size(data,1)),:);
cov_train = cov(data);
[evec, eval_matrix] = eig(cov_train);
eval_vector = diag(eval_matrix);
[evals, index] = sort(eval_vector, 'descend');
PCAVal = evals(1:m);
index_select = index(1:m);
PCAMat = evec(:,index_select);

end