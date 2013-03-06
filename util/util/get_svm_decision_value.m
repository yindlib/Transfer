function yscore = get_svm_decision_value(W, b, X)

% 2* (W'*E +  b*ones(1, size(E, 2)) > 0) -1;
yscore = [ W; b ]' * [ X'; ones(1, size(X, 1)) ];
yscore = yscore';
