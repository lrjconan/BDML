function [triplets, pairTgt, pairImp] = create_k_constraints(X, Y, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Create Constraints for Large Margin Metric Learning
%
%
% In
%   X (N x D) : data matrix (one sample per row)
%   Y (N x 1) : label matrix
%   k         : parameter class
%   
% Out
%   triplets (m  x 3) : m triplets of indices (center, target, imposter)
%   pairTgt  (pT x 2) : pT pairs of target neighbors
%   pairImp  (pI x 2) : pI pairs of imposter neighbors
%
% Author:
%   Renjie Liao
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k2          = k^2;
num         = size(X, 1);
classVal	= unique(Y);
classNum 	= numel(classVal);

pairTgt 	= zeros(num*k, 2);
pairImp 	= zeros(num*k, 2);
triplets 	= zeros(num*k2, 3);

idxTgtTree  = cell(classNum, 1);
idxImpTree  = cell(classNum, 1);
tgtTree     = cell(classNum, 1);
impTree 	= cell(classNum, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build KD-Tree
for i = 1 : classNum
    idxTgtTree{i}   = find(Y == classVal(i));
    tgtTree{i}      = KDTreeSearcher(X(idxTgtTree{i}, :), 'distance', 'euclidean');
    
    idxImpTree{i}   = find(Y ~= classVal(i));
    impTree{i}      = KDTreeSearcher(X(idxImpTree{i}, :), 'distance', 'euclidean');    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Search Neighbors

for i = 1 : num
    if mod(i, 1000) == 1
        tic;
    end
    
    idxClass        = find(Y(i) == classVal);
    
    %% search k target neighbors    
    idxCur          = knnsearch(tgtTree{idxClass}, X(i, :), 'k', k+1);    
    idxTgt          = idxTgtTree{idxClass}(idxCur(2:end));
            
    %% search k imposter neighbors    
    idxCur          = knnsearch(impTree{idxClass}, X(i, :), 'k', k);        
    idxImp          = idxImpTree{idxClass}(idxCur);
    
    %% construct output
    [idxX, idxY]                    = meshgrid(1:k, 1:k);    
    pairTgt((i-1)*k+1 : i*k, 1)     = i;
    pairTgt((i-1)*k+1 : i*k, 2)     = idxTgt;
    pairImp((i-1)*k+1 : i*k, 1)     = i;
    pairImp((i-1)*k+1 : i*k, 2)     = idxImp;
    triplets((i-1)*k2+1 : i*k2, 1)  = i;
    triplets((i-1)*k2+1 : i*k2, 2)  = idxTgt(idxX(:));
    triplets((i-1)*k2+1 : i*k2, 3)  = idxImp(idxY(:));  
    
    if mod(i, 1000) == 0
        time = toc;
        fprintf('Finish %03d thousand samples cost: %6.3f!\n', i/1000, time);
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

