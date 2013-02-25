function [ave_F1, ave_precision, ave_recall] =evaluate_performance(predict_labels, true_label)


F1_score=[];
precision_score = [];
recall_score = [];
for i=1:size(predict_labels,1)
    predict_labels_this = predict_labels(i,:);
    true_label_this = true_label(i,:);
    
    true_pos= find(true_label_this ==1);
    true_neg= find(true_label_this ~=1);
    
    predict_labels_pos = predict_labels_this(true_pos);
    ww= find(predict_labels_pos==1);
    
    if length(true_pos)==0
        disp('1')
        precision = 1;
    else
        precision = length(ww)/length(true_pos);
    end
    
    predict_labels_neg = predict_labels_this(true_neg);
    ww= find(predict_labels_neg~=1);
    
    if length(true_neg)==0
        disp('2')
        recall=1;
    else
        recall = length(ww)/length(predict_labels_neg);
    end
    
    F1_score= [F1_score, sqrt(precision*recall)];
    precision_score = [precision_score, precision];
    recall_score = [recall_score, recall];
end

ave_F1=mean(F1_score);
ave_precision = mean(precision_score);
ave_recall = mean(recall_score);