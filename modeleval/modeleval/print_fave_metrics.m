function s = print_fave_metrics(perf)

s = sprintf(' Acc=%5.5g  Prec=%5.5g   Rec=%5.5g    F1=%5.5g    GM=%5.5g   MCC=%5.5g  AUROC=%5.5g  AUPRC=%5.5g', perf.a, perf.p, perf.r, perf.f1, perf.gmean, perf.mcc, perf.auroc, perf.auprc);