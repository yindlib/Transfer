function [ auprc ] = perf_auprc(y, yscore)

[x, y, t, auprc] = perfcurve(y, yscore, 1, 'xCrit', 'reca', 'yCrit', 'prec');
