function perf = perf_fave_metrics(y, yhat, yscore)

perf.a = perf_accuracy(y, yhat);
[ perf.f1, perf.p, perf.r ] = perf_f1(y, yhat);
perf.mcc = perf_mcc(y, yhat);
[ perf.auroc, perf.opt_fp, perf.opt_tp ] = perf_auroc(y, yscore);
[ perf.auprc ] = perf_auprc(y, yscore);
[ perf.gmean, ~, ~ ] = perf_gmean(y, yhat);

% perf.s = sprintf(' Acc=%5.5g  Prec=%5.5g   Rec=%5.5g    F1=%5.5g    GM=%5.5g   MCC=%5.5g  AUROC=%5.5g  AUPRC=%5.5g', perf.a, perf.p, perf.r, perf.f1, perf.gmean, perf.mcc, perf.auroc, perf.auprc);