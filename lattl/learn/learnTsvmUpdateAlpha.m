function [alpha, alpha_b] = learnTsvmUpdateAlpha(C, e, s, w, b)

[ne,de] = size(e);
% x = [e;e];
% nx = 2 * ne;
% y = zeros(nx,1);
% y(1 : ne,1) = 1;
% y((ne + 1) : nx,1) = -1;
% 
% alpha = zeros(de,1);
% alpha_b = 0;
% 
% for i = 1 : nx
%     f = y(i) * (x(i,:) * w + b);
%     if (f < s)
%         alpha = alpha + C * y(i) * x(i,:)';
%         alpha_b = alpha_b + C * y(i);
%     end
% end

% Taha
a = sum([e, ones(ne, 1)].*repmat(( (e*w+b) < s), 1, de+1), 1) + ...
    sum(-[e, ones(ne, 1)].*repmat(( -(e*w+b) < s), 1, de+1), 1);
a = a'*C;
alpha = a(1:end-1);
alpha_b = a(end);
