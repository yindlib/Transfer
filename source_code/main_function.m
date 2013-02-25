function predict_labels = main_function(source_data, target_data, parameters)

% label_set=[source_data.label];
% instance_index = choose_instances(label_set);
% source_data= source_data(sum(instance_index)>0);
all_instance = [source_data.instance];
all_instance = all_instance - mean(all_instance, 2)*ones(1, size(all_instance, 2));
label_set=[source_data.label];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_source_data = length(source_data);
target_instance = [target_data.instance];
target_instance = target_instance - mean(target_instance, 2)*ones(1, size(target_instance, 2));
num_target_data = length(target_data);
target_label = [target_data.label];

% Initialization
% PCA to initialize a, e, Phi and Psi

% % Rotate the data
% t= 270/180*pi;
% R = [cos(t) -sin(t); sin(t) cos(t)];
% all_instance(1:2, :) = R*all_instance(1:2, :);

figure
hold on
scatter(all_instance(1, 1:300), all_instance(2, 1:300), 'b')
scatter(all_instance(1, 301:600), all_instance(2, 301:600), 'g')

figure
hold on
scatter(target_instance(1, 1:300), target_instance(2, 1:300), 'b')
scatter(target_instance(1, 301:600), target_instance(2, 301:600), 'g')

Phi = pca(parameters.r, all_instance');
Psi = pca(parameters.r, target_instance');
% Temporary
if size(all_instance, 1) == parameters.r
    Phi = eye(parameters.r);
    Psi = Phi;
end

A = Phi'*all_instance;
E = -Psi'*target_instance;
% SVM on a and e to initialize w, b
% For each w_j
W = zeros(parameters.r, size(label_set, 1));
b = zeros(1, size(label_set, 1));
par=sprintf('-c %f -q', parameters.C2);
for j = 1:size(label_set, 1)
    model = svmtrain(label_set', A', par);
    
    W(:,j) = model.SVs' * model.sv_coef;
    b(:,j) = -model.rho;
    if model.Label(1) == -1
        W(:,j) = -W(:,j);
        b(:,j) = -b(:,j);
    end
end

% Getting the performance of the PCA + SVM
trn_error = sum( abs( 2*([W; b]'*[A; ones(1, size(A, 2))] >0)-1- label_set ) )/length(label_set)/2;
tst_error = sum( abs( 2*([W; b]'*[E; ones(1, size(E, 2))] >0) -1- target_label ) )/length(target_label)/2;
R2 = (1/300)*(sqrt(trace(A'*A)));
fprintf('PCA+SVM: %f,\t%f,\t%f\n', trn_error, tst_error, R2)

W_old = W;
for i =1:parameters.max_iter
    % calculate \Phi & \Psi
    fprintf('%d\t', i)
    fprintf('Basis...')
    Phi = l2ls_learn_basis_dual(all_instance, A, numel(Phi), Phi);
%     disp(objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters))
    Psi = l2ls_learn_basis_dual(target_instance, E, numel(Psi), Psi);
%     disp(objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate w, b
    fprintf('Classifiers...');
    for ii=1:size(label_set, 1)
        [W(:, ii), b(:, ii)] = newLearn(W(:, ii), b(:, ii), A', 2*label_set(ii, :)'-1, E', parameters);
    end
%     disp(objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Obatin A
    fprintf('Vectors...\n');
    target_labels = 2* (W'*E +  b*ones(1, size(E, 2)) > 0) -1;
    for ii=1:num_source_data
        [ A(:, ii), ~, ~,~ ] = aLearn( W', b', Phi, label_set(:, ii), all_instance(:, ii), parameters.C1, parameters.C2 );
    end
%     disp(objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters))
    for ii=1:num_target_data
        [ E(:, ii), ~, ~,~ ] = aLearn( W', b', Psi, target_labels(:, ii), target_instance(:, ii), parameters.C1, parameters.C3 );
    end
%     disp(objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters))
    
    if norm(W-W_old, 'fro')<parameters.epilson
        fprintf('\n The difference is: %f\n', norm(W-W_old, 'fro'));
        break;
    end
end

predict_labels = target_labels;

% Drawing the graph
figure
hold on
scatter(E(1, 1:300), E(2, 1:300), 'b')
scatter(E(1, 301:600), E(2, 301:600), 'g')

% Getting the performance of the PCA + SVM
trn_error = sum( abs( predict_labels- label_set ) )/length(label_set)/2;
tst_error = sum( abs( predict_labels- target_label ) )/length(target_label)/2;
R2 = (1/600)*(sqrt(trace(A'*A)) + sqrt(trace(E'*E)));
fprintf('LATTL: %f,\t%f,\t%f\n', trn_error, tst_error, R2)

% % Getting the performance of the PCA + SVM
% trn_error = sum( abs( 2*(w_i*[A; ones(1, size(A, 2))] >0)-1- label_set ) )/length(label_set)/2;
% tst_error = sum( abs( 2*(w_i*[E; ones(1, size(E, 2))] >0) -1- target_label ) )/length(target_label)/2;
% R2 = (1/600)*(sqrt(trace(A'*A)) + sqrt(trace(E'*E)));
% fprintf('PCA+SVM: %f,\t%f,\t%f\n', trn_error, tst_error, R2)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = objective2(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters)
% First term
out = 0.5*( norm(W, 'fro')^2+ norm(b)^2); 
% Second Terms
out = out + parameters.C1*( norm(all_instance - Phi*A, 'fro')^2 + parameters.beta*norm(Phi, 'fro')^2);
out = out + parameters.C1*( norm(target_instance - Psi*E, 'fro')^2 + parameters.beta*norm(Psi, 'fro')^2);
% Third Terms
xi = 1 - label_set .* (W'*A + repmat(b', 1, size(A, 2)));
out = out + parameters.C2*sum(sum((xi.*(xi>0))));
temp =  1-abs( W'*E + repmat(b', 1, size(E, 2)) );
out = out + parameters.C3*sum(sum( temp.*(temp > 0) ));

end

function out = objective(all_instance, target_instance, label_set, Phi, Psi, A, E, W, b, parameters)
% First term
out = 0.5*( norm(W, 'fro')^2+ norm(b)^2); 
% Second Terms
out = out + parameters.C1*( norm(all_instance - Phi*A, 'fro')^2 + parameters.beta*norm(Phi, 'fro')^2);
out = out + parameters.C1*( norm(target_instance - Psi*E, 'fro')^2 + parameters.beta*norm(Psi, 'fro')^2);
% Third Terms
% Third Terms
xi = 1 - label_set .* (W'*A + repmat(b', 1, size(A, 2)));
out = out + parameters.C2*sum(sum((xi.*(xi>0))));
xi1 =  1 - (W'*E + repmat(b', 1, size(E, 2)));
xi2 =  1 + (W'*E + repmat(b', 1, size(E, 2)));
out = out + parameters.C3*sum(sum( (xi1.*(xi1>0)) + (xi2.*(xi2>0))));
s = -0;
xi1 =  s - (W'*E + repmat(b', 1, size(E, 2)));
xi2 =  s + (W'*E + repmat(b', 1, size(E, 2)));
out = out - parameters.C3*sum(sum( (xi1.*(xi1>0)) + (xi2.*(xi2>0))));

end