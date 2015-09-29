% clear;clc;
% load('tmp_PCA_70_ball_NN_1.mat');

% load('tmp_embed_100_ball_NN_1');

feat_train_proj = feat_train;
feat_test_proj  = feat_test;
test_folder     = './exp/Kaggle_res_embed_100_ball_NN_1';

num_NN = 1;
file_names = dir(fullfile(test_folder, sprintf('*.mat')));
file_num = numel(file_names);

dim_feat    = 100;
best_M      = zeros(dim_feat);
best_acc    = 0;
best_cond   = 0;

for i = 37:5:file_num  
    load(fullfile(test_folder, file_names(i, 1).name));
    
    M             = X_save(1:dim_feat, 1:dim_feat);
    acc_test_BDML = test_KNN(feat_test_proj, label_test, feat_train_proj, label_train, eye(100), num_NN)*100;
    
    if acc_test_BDML > best_acc
        best_M      = M;
        best_cond   = cond(M);
        best_acc    = acc_test_BDML;
    end
    
    fprintf('File = %s || Best accuracy = %5.2f%% \n', file_names(i, 1).name, best_acc);
end
