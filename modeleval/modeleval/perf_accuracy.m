function a = perf_accuracy(y, yhat)

a = sum(y==yhat) / numel(y);