function partitioning = data_cv_kfold_unbal(labels, k)

frac = repmat(1/k, 1, k);
partitioning = data_split_unbal(labels, frac);

% uniq_lab = unique(labels);
% partitioning = zeros(size(labels));
% 
% for i=1:numel(uniq_lab)
%     partitioning(labels==uniq_lab(i)) = perf_kfold(labels(labels==uniq_lab(i)), k);
% end
