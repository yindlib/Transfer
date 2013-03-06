function partitioning = data_split_unbal(labels, frac)

uniq_lab = unique(labels);
partitioning = zeros(size(labels));

for i=1:numel(uniq_lab)
    partitioning(labels==uniq_lab(i)) = data_split(labels(labels==uniq_lab(i)), frac);
end
