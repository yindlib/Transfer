function [source_data, target_data] =  get_data_advance2(parameters)

cd '../source_data';

cd(parameters.data_set);

file_name_source = dir('*source*');

load(file_name_source.name);

source_data = data_store;


file_name_source = dir('*target*');

load(file_name_source.name);

target_data = data_store;

cd '../../source_code';


num_source_data = length(source_data);
index=randperm(num_source_data);

source_data = source_data(index);

source_data = source_data(1:ceil(parameters.ratio*num_source_data));

label_set = [source_data.label];

label_set_ind = (label_set==1);

label_set_ind = (sum(label_set_ind,2)>0);

for i=1:length(source_data)
    temp = source_data(i).label;
    source_data(i).label = temp(label_set_ind);    
end

for i=1:length(target_data)
    temp = target_data(i).label;
    target_data(i).label = temp(label_set_ind);    
end





% label_set = [source_data.label];
% 
% aa=0;
% instance_index= [];
% for i=1:size(label_set, 1)
%     label_temp = label_set(i,:);
%     ww1=find(label_temp==1);
%     ww2=find(label_temp~=1);
%     pos_ind = ww1;
%     neg_ind=  ww2;
%     rerank_pos = randperm(length(pos_ind));
%     pos_ind = pos_ind(rerank_pos);
%     rerank_neg = randperm(length(neg_ind));
%     neg_ind = neg_ind(rerank_neg);
%     
%     if length(pos_ind)>parameters.train_data
%         pos_ind = pos_ind(1:parameters.train_data);
%     
%     end
%     if length(neg_ind)>parameters.train_data
%         neg_ind = neg_ind(1:parameters.train_data);
%     end
%      temp = zeros(1, size(label_set, 2));
%     temp([pos_ind, neg_ind]) = 1;
%     if sum(temp)==0
%        aaa=0; 
%     end
%     
%      instance_index=[instance_index; temp];
% end
% 
% 
% instance_index = (sum(instance_index)>0);
% 
% source_data=source_data(instance_index);


% 
% %%%%%%%normalize
if parameters.normalize==1
source_instance = [source_data.instance];
source_instance_ind =max(abs(source_instance'));
ww = find(source_instance_ind ==0);
source_instance_ind(ww)=1;
for i=1:length(source_data)
    source_data(i).instance = (source_data(i).instance)./ repmat(source_instance_ind', 1, size(source_data(i).instance, 2));
    
end
for i=1:length(target_data)
    target_data(i).instance = (target_data(i).instance)./ repmat(source_instance_ind', 1, size(target_data(i).instance, 2));
    
end


end









