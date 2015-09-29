clear; clc;

%% Load Dataset
nameDatabase    = 'kaggle_embed_100_ball_NN_1';
pathData        = ['.\data\' nameDatabase '.mat'];
load(pathData);

%% Setting Parameters
para.K                      = 1.0e+3;       % distortion bound
para.quiet                  = 0;
para.mu                     = 0.0;       % margin parameter 
para.upb                    = 1.0e-2;       % upper bound of objective function value
para.lambda_distortion      = 1.0e+1;   
% para.flagDiag               = 'diagonal';
para.flagDiag               = 'full';

para.solverMWU.delta        = 1.0e-1;       % delta for ORACLE
para.solverMWU.epsilon      = 0.5;          % epsilon for Multiplicative Update method
para.solverMWU.iterMaxEig   = 1000;
para.solverMWU.tolEig       = 1.0e-9;
para.solverMWU.quiet        = 1;
para.solverMWU.iterMax      = 1000000;
para.solverMWU.traceBound   = 1.0e+1;
para.solverMWU.rho          = 1.0e+5;
para.solverMWU.ell          = 1.0e+5;          % epsilon for Multiplicative Update method
para.solverMWU.dispIter     = 10;
para.solverMWU.saveIter     = 100;
para.solverMWU.savePath     = './exp/Kaggle_res_embed_100_ball_NN_1';

%%
numNN       = 1;
dim_feat    = size(feat_train, 2);

const = max(abs(feat_train(:)));
feat_train = feat_train./const;
feat_test = feat_test./const;

%% Dimension Reduction
%{
dim_PCA             = 50;
[~, projection_mat] = PCA(feat_train, dim_PA);
feat_train_copy     = feat_train * projection_mat;
feat_test_copy      = feat_test * projection_mat;
%}

%% Create or Load Constraints
% cons_name = 'kaggle_embed_100_ball_NN_1.mat';
% cons_path = './data/';
% load(fullfile(cons_path, cons_name));

% [triplets, pairTgt, pairImp] = create_k_constraints(feat_train_copy, label_train, numNN);
% [triplets, pairTgt, pairImp] = create_ball_constraints(feat_train_copy, label_train, numNN);

%% Bounded Distortion Metric Learning with Triplet Constraints
[M_BDML, info_BDML] = tBDML(feat_train, label_train, pairTgt, triplets, para); 

if ~info_BDML.fail
    acc_test_BDML  = test_KNN(feat_test, label_test, feat_train, label_train, M_BDML, numNN);
    fprintf('tBDML: Test Accuracy = %5.2f%% \n', acc_test_BDML*100);
else
	fprintf('tBDML Solver Failed!\n');
end

acc_test_EUC = test_KNN(feat_test, label_test, feat_train, label_train, eye(dim_feat), numNN); 
fprintf('EUC : Test Accuracy = %5.2f%% \n', acc_test_EUC*100);

%% save result
%{
savePath = '.\exp\UCI\';
if ~isdir(savePath)
    mkdir(savePath);
end

save(fullfile(savePath, [nameDatabase '_t_BDML_res.mat']), 'best_result');
fprintf('BDML: Best Mean Test Error is %5.2f%% || Std is %2.1f%% \n', ...
        best_result.minErr, best_result.stdErr);
%}
