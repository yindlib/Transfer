clc
clear

addpath /home/taha/work/libsvm

% Parameter Settings
parameters.C = 1;
parameters.C1 = 0.1;
parameters.C2 = parameters.C;
parameters.C3 = 0.1;
parameters.beta= 0.0001;
parameters.epsilon = 0.01;
parameters.r = 100;
parameters.s = 0;
parameters.opt_algo_a = 'interior-point-convex';
parameters.large_scale_a = 'off';
parameters.opt_algo_wb = 'interior-point-convex';
parameters.large_scale_wb = 'off';
parameters.wbc = false;
parameters.data_set = 'synthetic';
parameters.mosek_path = 'C:\Program Files\Mosek\6\toolbox\r2009b';
parameters.ratio = 0.5;
parameters.tratio = 0.1;
parameters.normalize=1;
parameters.max_iter_inner = 5;
parameters.max_iter_outer = 5;

% Get Data
[source, target] =  get_data_advance2(parameters);

source_data.instance = [source.instance]';
source_data.label = [source.label]';
source_data.instance = source_data.instance - mean(source_data.instance, 2)*ones(1, size(source_data.instance, 2));


target_data.instance = [target.instance]';
target_data.label = [target.label]';
target_data.instance = target_data.instance - mean(target_data.instance, 2)*ones(1, size(target_data.instance, 2));


auxiliary_data.instance = [];

% Algorithm Test
tic
[ W, b, PhiZ, a, e, preds ] = main_function( source_data, target_dataparameters );
toc
misclassified_instances = sum(preds ~= target_data.label);
[ave_F1, ave_precision, ave_recall]  = evaluate_performance(preds, 2*target_data.label-1);
% [avgTPR, avgFPR, avgACC, avgSPC, avgPPV, avgNPV, avgFDR, avgMCC, avgF1] = evaluate_performance(preds, target_data.label);
save('Results.mat', 'ave_F1', 'ave_precision', 'ave_recall')