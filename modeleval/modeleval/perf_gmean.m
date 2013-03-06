function [ gm, p, r ] = perf_gmean(y, yhat)

p = perf_precision(y, yhat);
r = perf_recall(y, yhat);

gm = sqrt(p * r);
