function [ auroc, opt_fp, opt_tp ] = perf_auroc(y, yscore)

%auroc = scoreAUC(y, yscore);
[x, y, t, auroc, opt] = perfcurve(y, yscore, 1, 'xCrit', 'FPR', 'yCrit', 'TPR');
opt_fp = opt(1);
opt_tp = opt(2);

