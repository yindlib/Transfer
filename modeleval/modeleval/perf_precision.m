function p = perf_precision(y, yhat)

p = sum((y==1) & (yhat==1)) / sum(yhat==1);