function partitioning = data_split(labels, frac)

grp = [];
for fi=1:(length(frac)-1)
    grp = [ grp; repmat(fi, max(1, floor(frac(fi) * length(labels))), 1) ];
end
grp = [ grp; repmat(length(frac), length(labels) - length(grp), 1) ];

inds = randperm(length(labels));

[~, idx] = sort(inds);
partitioning = grp(idx);
