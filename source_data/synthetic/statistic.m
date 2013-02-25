
file_name_source = dir('*source*');

load(file_name_source.name);

source_data = data_store;


file_name_source = dir('*target*');

load(file_name_source.name);

target_data = data_store;



source_num=length(source_data);
target_num = length(target_data);


source_instance = [source_data.instance];

source_ind = (source_instance~=0);

sparsity_source =mean(mean(source_ind));



target_instance = [target_data.instance];

target_ind = (target_instance~=0);

sparsity_target=mean(mean(target_ind));