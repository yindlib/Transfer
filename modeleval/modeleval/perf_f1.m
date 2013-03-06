function [ f1, p, r ] = perf_f1(y, yhat)

p = perf_precision(y, yhat);
r = perf_recall(y, yhat);

f1 = 2 * (p * r) / (p + r);
