function [ scaled ] = scale_data_unit_var(data)

m = mean(data, 1);
s = std(data, 1);
scaled = (data - repmat(m, size(data, 1), 1)) ./ repmat(s, size(data,1), 1);

%scaled = (data - repmat(mean(data,1),size(data,1),1)) * spdiags(1./(std(data,1))',0,size(data,2),size(data,2));