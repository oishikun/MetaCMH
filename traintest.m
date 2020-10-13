label(269649,:) = sum(label,1);
label = label';
label = sortrows(label, 269649);
label = label';
label(269649,:) = [];
label(:, 1:2) = [];
train_ind = zeros(269648,1);
test_ind = zeros(269648,1);

num_test = 30;
for i = 1:79
   if i < 35
       num_train = 50;
   elseif i < 70
       num_train = 100;
   else
       num_train = 1000;
   end
   
   count_j = 0;
   for j = 1:269648
       if label(j, i) == 1 
           train_ind(j,1) = 1;
           count_j = count_j + 1;
           if count_j == num_train
               break;
           end
       end
   end
   
   count_k = 0;
   for k = 1:269648
       if label(k, i) == 1 
           test_ind(k,1) = 1;
           count_k = count_k + 1;
           if count_k == num_test
               break;
           end
       end
   end
end

retrieval_ind = (test_ind == 0);

train_L = label(train_ind,:);
test_L = label(test_ind,:);
retrieval_L = label(retrieval_ind,:);

train_x = image(train_ind,:,:,:);
test_x = image(test_ind,:,:,:);
retrieval_x = image(retrieval_ind,:,:,:);

train_y = text(train_ind,:,:,:);
test_y = text(test_ind,:,:,:);
retrieval_y = text(retrieval_ind,:,:,:);

save('NUS-wide79splited.mat','train_L','train_x','train_y','test_L','test_x','test_y','retrieval_L','retrieval_x','retrieval_y','-v7.3')


