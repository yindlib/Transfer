%  alpo_main
parameters.data_set = 'synthetic';
parameters.mosek_path = 'C:\Program Files\Mosek\6\toolbox\r2009b';
parameters.C1=32;
parameters.C2=4;
parameters.beta= 0.05;
parameters.s=20;
parameters.epilson = 0.01;
parameters.k=10; % number of clusters
parameters.max_iter=30;
parameters.train_data=40;
parameters.ratio = 0.2;
parameters.normalize=1;
[source_data, target_data] =  get_data_advance2(parameters);

% [source_data, target_data] = pca_preprocess(source_data, target_data);

[predict_labels, clusters, cluster_assignment] = main_function_alpo_newinit2(source_data, target_data, parameters);


[ave_F1, ave_precision, ave_recall] =evaluate_performance(predict_labels, [target_data.label]);