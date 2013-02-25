%  alpo_main
clc
clear all
close all

parameters.data_set = 'synthetic';
parameters.C1=0.1;
parameters.C2=0.1; % 10
parameters.C3=100;   % 0.3
parameters.beta= 0.00001;
parameters.r=6;
parameters.epilson = 0.01;
parameters.max_iter=3;
parameters.ratio = 1;
parameters.normalize=0;
[source_data, target_data] =  get_data_advance2(parameters);


C2 = 10; %logspace(-2, 2, 8);
C3 = 0.3; %logspace(-2, 2, 8);
Results = zeros(8);
for i=1:length(C2)
    for j = 1:length(C3)
        parameters.C2=C2(i); % 10
        parameters.C3=C3(j);   % 0.3
        predict_labels = main_function(source_data, target_data, parameters);
        
        [ave_F1, ave_precision, ave_recall] =evaluate_performance(predict_labels, [target_data.label]);
        Results(i, j) = ave_F1;
    end
end