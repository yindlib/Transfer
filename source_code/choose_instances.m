function instance_index = choose_instances( label_set)

num_instance=size(label_set, 2);
instance_index = [];
for i=1:size(label_set, 1)
    label_temp = label_set(i,:);
    ww1=find(label_temp==1);
    ww2=find(label_temp~=1);
    pos_ind = ww1;
    neg_ind=  ww2;
    
    rerank_neg = randperm(length(neg_ind));
    neg_ind = neg_ind(rerank_neg);
    if length(pos_ind)<length(neg_ind)
        neg_ind = neg_ind(1:length(pos_ind));
    end
    all_ind = [pos_ind, neg_ind];
    temp=zeros(1, num_instance);
    temp(all_ind)=1;
    if sum(temp)==0
        aaa=0;
    end
    instance_index=[instance_index; temp];
    
end