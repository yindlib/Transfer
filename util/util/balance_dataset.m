function keepidx = balance_dataset(y)

idx = randperm(size(y,1));
posidx = idx(y(idx)==1);
negidx = idx(y(idx)~=1);

nkeep = min(length(posidx), length(negidx));

keepidx = sort([ posidx(1:nkeep) negidx(1:nkeep) ]);
