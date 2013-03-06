function [ auroc, opt ] = perf_auc(y, yscore)

%auroc = scoreAUC(y, yscore);
[x, y, t, auroc, opt] = perfcurve(ytrue, yprob, 1, 'xCrit', 'FPR', 'yCrit', 'TPR');
