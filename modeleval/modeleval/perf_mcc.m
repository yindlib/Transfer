function mcc = perf_mcc(y, yhat)

tp = sum((y==1) & (yhat==1));
fp = sum(yhat==1) - tp;
tn = sum((y~=1) & (yhat~=1));
fn = sum(yhat~=1) - tn;

mcc_denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
if mcc_denom == 0
    mcc_denom = 1;
end
mcc = (tp * tn - fp * fn) / mcc_denom;
