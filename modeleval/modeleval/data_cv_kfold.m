function partitioning = data_cv_kfold(labels, k)

frac = repmat(1/k, 1, k);
partitioning = data_split(labels, frac);

% foldsizes = repmat(floor(size(labels,1) / k), 1, k);
% foldsizes = foldsizes + [ ones(1, mod(size(labels,1), k)) zeros(1, k - mod(size(labels,1), k)) ];
% 
% fold = [];
% for i=1:k
%     fold = [ fold repmat(i, 1, foldsizes(i)) ];
% end
% 
% idx = randperm(size(labels,1));
% 
% [~, idx] = sort(idx);
% partitioning = grp(idx);
