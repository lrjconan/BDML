function [triplets, pairTgt, pairImp] = create_ball_constraints(X, Y, k)
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

numTgt      = 0;                       
numImp      = 0;
numTpt      = 0;
num         = size(X, 1);
classVal    = unique(Y);
classNum    = numel(classVal);
subIdx      = cell(classNum, 1);
subtree     = cell(classNum, 1);
idxTgt      = cell(num, 1);             % index of target neighbors
idxImp      = cell(num, 1);             % index of imposter neighbors
idxTpt      = cell(num, 1);             % index of triplets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build KD-Tree
tree = KDTreeSearcher(X, 'distance', 'euclidean');
for i = 1 : classNum
    subIdx{i}   = find(Y == classVal(i));
    subtree{i}  = KDTreeSearcher(X(subIdx{i}, :), 'distance', 'euclidean');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Search Neighbors

for i = 1 : num
    if mod(i, 1000) == 1
        tic;
    end    
    
    %% search k target neighbors
    idxClass            = find(Y(i) == classVal);
    [idxCur, distCur]   = knnsearch(subtree{idxClass}, X(i, :), 'k', k+1);        
    idxTgt{i}           = subIdx{idxClass}(idxCur(2:end));
    
    %% search imposter neighbors
    idxCur              = cell2mat(rangesearch(tree, X(i, :), distCur(end)));            
    idxImp{i}           = setdiff(idxCur(:), [idxTgt{i}; i]);        
    
    %% triplet
    numX                = numel(idxTgt{i});
    numY                = numel(idxImp{i});
    numT                = numX*numY;
    [idxX, idxY]        = ind2sub([numX numY], 1:numT);
    idxTpt{i}           = [idxTgt{i}(idxX(:)) idxImp{i}(idxY(:))];
    
    numTgt              = numTgt + numX;
    numImp              = numImp + numY;
    numTpt              = numTpt + numT;
    
    if mod(i, 1000) == 0
        time = toc;
        fprintf('Finish %03d thousand samples cost: %6.3f!\n', i/1000, time);
    end     
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construct Output

cntTgt      = 1;
cntImp      = 1;
cntTpt      = 1;
triplets    = zeros(numTpt, 3);
pairTgt     = zeros(numTgt, 2);
pairImp     = zeros(numImp, 2);

for i = 1 : num
    numTmp = size(idxTgt{i}, 1);
    if numTmp > 0 && ~isempty(idxTgt{i})
        pairTgt(cntTgt : cntTgt + numTmp - 1, 1) = i;
        pairTgt(cntTgt : cntTgt + numTmp - 1, 2) = idxTgt{i};
        cntTgt = cntTgt + numTmp;
    end
    
    numTmp = size(idxImp{i}, 1);
    if numTmp > 0 && ~isempty(idxImp{i})
        pairImp(cntImp : cntImp + numTmp - 1, 1) = i;
        pairImp(cntImp : cntImp + numTmp - 1, 2) = idxImp{i};
        cntImp = cntImp + numTmp;
    end
    
    numTmp = size(idxTpt{i}, 1);  
    if numTmp > 0 && ~isempty(idxTpt{i})
        triplets(cntTpt : cntTpt + numTmp - 1, 1) = i;
        triplets(cntTpt : cntTpt + numTmp - 1, 2) = idxTpt{i}(:, 1);
        triplets(cntTpt : cntTpt + numTmp - 1, 3) = idxTpt{i}(:, 2);
        cntTpt = cntTpt + numTmp;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

