function r = perf_recall(y, yhat)

r = sum((y==1) & (yhat==1)) / sum(y==1);