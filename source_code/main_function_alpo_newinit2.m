function [predict_labels, clusters_label, cluster_assignment, time] = main_function_alpo_newinit2(source_data, target_data, parameters)
% a different initialization method from the previous one.
%current
%Higher Level Representation
%initialize



label_set=[source_data.label];
instance_index = choose_instances(label_set);
% instance_index = ones(size(instance_index,1), size(instance_index,2));
all_index = find(sum(instance_index)>0);
instance_index=instance_index(:, all_index);
source_data= source_data(all_index);
all_instance = [source_data.instance];
label_set=[source_data.label];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_source_data = length(source_data);

%  phi = rand(size(all_instance,1),parameters.s)-0.5;
% 	phi = phi - repmat(mean(phi,1), size(phi,1),1);
%     phi = phi*diag(1./sqrt(sum(phi.*phi)));
%
%
% new_instances_init = all_instance'* phi*inv(phi'*phi + parameters.beta* eye(size(phi,2)));
% new_instances_init = new_instances_init';
%
%
% A = new_instances_init;
%

if parameters.s>size(all_instance,1)
    A = all_instance;
    A=[A; randn(parameters.s-size(all_instance,1), size(all_instance,2))];
end

for i =1:parameters.max_iter
    tic;
    A_old=A;
    %calculate \Phi
    %       B = l2ls_learn_basis_dual(X, S, l2norm, Binit)
    fprintf('Learning basis............. \n')
    phi = l2ls_learn_basis_dual(full(all_instance), A, 1);
    time1=toc;
    fprintf('Done! time:  %f\n', time1);
    %calculate w, b
    W=[];
    b=[];
    fprintf('Learning Classifiers............***');
    for ii=1:size(label_set, 1)
        fprintf('%c%c%c%c', 8,8,8);
        fprintf('%3d', ii);
        temp_ind =  instance_index(ii,:);
        ww = find(temp_ind>0);
        this_label = label_set(ii,:);
        this_label=this_label(ww);
        this_instance = A(:, ww);
        
        par=sprintf('-c %f', parameters.C2);
        
        if length(unique(this_label))~=2
            aaa=0;
        end
        
        model = train(double(full(this_label)'), sparse([this_instance',ones(size(this_instance,2),1)]), par);
        %
        if model.nr_class~=2
            aaa=0;
        end
        w_i=model.w;
        if length(this_label)==0
            disp('Problem!!');
            pause;
        end
        
        if this_label(1)~=1
            W=[W,  (-w_i(1:(end-1)))'];
            b=[b, -w_i(end)];
        else
            W=[W,  (w_i(1:(end-1)))'];
            b=[b, w_i(end)];
        end
        
        %          par=sprintf('-t 0 -c %f', parameters.C2);
        %          model = svmtrain(full(this_label'), full(this_instance'), par);
        %          w_i = sum(repmat((model.sv_coef)', size(model.SVs, 2), 1).*(model.SVs)',2);
        %
        %          if this_label(1)==1
        %          W=[W, w_i];
        %          b=[b, -model.rho];
        %          else
        %            W=[W, -w_i];
        %          b=[b, model.rho];
        %          end
    end
    time2=toc;
    fprintf('\nDone, time:%f \n', time2-time1);
    save temp.mat
    % obatin A
    addpath(parameters.mosek_path);
    A= [];
    %        K=phi'*phi+parameters.beta*eye(size(phi, 2));
    %        K =  [parameters.C1*K, zeros(size(K,1), size(label_set, 2)); zeros(size(label_set, 1),size(K,2) ), zeros(size(label_set, 1), size(label_set, 2))];
    %        K=2*K;
    
    
    new_instances_init = all_instance'* phi*inv(phi'*phi + parameters.beta/2* eye(size(phi,2)));
    new_instances_init = new_instances_init';
    fprintf('Outer iteration: %d, Inner total: %d,', i, num_source_data);
    fprintf(' Inner iteration: %5d',num_source_data );
    
    K_store = W'*W;
    
    % instance_index_back =instance_index;
    % instance_index = ones(size(instance_index,1), size(instance_index,2));
    for ii=1:num_source_data
        %        fprintf('num_of_instance: %d\n', i);
        fprintf('%c%c%c%c%c%c', 8,8,8,8,8,8);
        fprintf('%5d ', ii);
        temp_ind = instance_index(:, ii);
        temp_ind=(temp_ind>0);
        
        this_label = label_set(:,ii);
        this_label = 2*this_label-1;
        
        this_label =  this_label(temp_ind);
        W_select = W(:, temp_ind);
        
        
        %%%%%%%%%%%%%%%%%%%
        %        C3=parameters.C2/length(this_label)/parameters.C1/parameters.beta;
        C3=parameters.C2/parameters.C1/parameters.beta;
        K  = K_store(temp_ind, temp_ind);
        K = K.*(this_label*this_label');
        
        f=(b(temp_ind))'.*this_label -1;
        
        
        lb=zeros(sum(temp_ind), 1);
        ub = C3* ones(sum(temp_ind), 1);
        alpha = quadprog(K,f,[],[],[],[],lb,ub);
        
        sol2= repmat(this_label',size(W_select,1),1).*W_select;
        sol2= sum(repmat(alpha',size(W_select,1),1).*W_select,2);
        sol1= new_instances_init(:,ii);
        
        dev = sol2 - sol1;
        sol_set = repmat(sol1, 1, 50)+ repmat(linspace(0,1, 50), size(sol1,1),1).*repmat(dev, 1, 50);
        
        term1 = phi * sol_set;
        term1 =  sum((repmat(all_instance(:,ii), 1, size(term1, 2)) - term1).^2);
        
        term1 = parameters.C1*(term1+parameters.beta*sum(sol_set.^2));
        
        temp=W_select'*sol_set + repmat((b(temp_ind))',1,size(sol_set,2));
        temp = repmat(this_label,1,size(temp,2)).*temp;
        
        temp =1-temp;
        ind=(temp>0);
        xi = temp.*ind;
        %        mean_xi = sum(xi)/size(temp,1);
        mean_xi = sum(xi);
        
        output =  term1+parameters.C2*mean_xi;
        
        [aa,ww]=min(output);
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        A=[A, sol_set(:,ww)];
        
    end
    %    instance_index= instance_index_back;
    fprintf('\n');
    time3=toc;
    fprintf('Done, time:%f \n', time3-time2);
    rmpath(parameters.mosek_path);
    
    if sum(sum((A-A_old).^2))<parameters.epilson
        fprintf('\n The difference is: %f\n', sum(sum((A-A_old).^2)));
        
        break;
    end
end
time = toc;

predict_labels = [];
clusters= [];
cluster_assignment = [];

target_instance =[target_data.instance];
fprintf('\n For target data ...............');

new_instances = target_instance'* phi*inv(phi'*phi + parameters.beta* eye(size(phi,2)));
new_instances = new_instances';



output  = W'*new_instances + repmat(b', 1, size(new_instances,2));
predict_labels = (output>0);


%KMeans
distance_store=[];
class_store=[];
Centres_store=[];

for i=1:10
    [Classes,Centres,FinalDistance]=dcKMeans(new_instances', parameters.k);
    distance_store=[distance_store, sum(FinalDistance)];
    class_store=[class_store,Classes];
    Centres_store = [Centres_store, {Centres}];
end
[u,v]=min(distance_store)

fprintf('done \n');
cluster_assignment=class_store(:,v);
center_instance =  Centres_store{v};

clusters_label  = W'*center_instance' + repmat(b', 1, size(center_instance,1));

clusters_label = (clusters_label>0);
