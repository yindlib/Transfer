function avg = perf_avg_fave_metrics(perfs)

avg.a = mean([ perfs.a ]);
avg.f1 = mean([ perfs.f1 ]);
avg.gmean = mean([ perfs.gmean ]);
avg.p = mean([ perfs.p ]);
avg.r = mean([ perfs.r ]);
avg.mcc = mean([ perfs.mcc ]);
avg.auroc = mean([ perfs.auroc ]);
avg.opt_fp = mean([ perfs.opt_fp ]);
avg.opt_tp = mean([ perfs.opt_tp ]);
avg.auprc = mean([ perfs.auprc ]);

