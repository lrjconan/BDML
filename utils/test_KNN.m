function [acc, yTeHat] = test_KNN(xTe, yTe, xTr, yTr, M, k)
%   test Large Margin Nearest Neighbor with learned metric M
%
%   
%
%
%   X (N x D) : data matrix (one sample per row)
%   Y (N x 1) : label matrix
%   k         : number of neighbors

%% 
numTe   	= size(xTe, 1);
numTr   	= size(xTr, 1);
yTr         = yTr(:)';

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform Cholesky Decomposition
U       	= chol(M);
xTrU     	= xTr*U';
xTeU     	= xTe*U';
tree    	= KDTreeSearcher(xTrU, 'distance', 'euclidean');
labelTe 	= zeros(numTe, 1);

%% Compute Index of Neighborhood
for i = 1 : numTe   
    [tmpIdx, tmpDist] 	= knnsearch(tree, xTeU(i, :), 'k', k);    
    labelTe(i)          = mode(yTr(tmpIdx));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute Distance Directly
dist        = repmat(diag(xTr*M*xTr'), 1, numTe) ...
            + repmat(diag(xTe*M*xTe')', numTr, 1) ...
            - 2*xTr*M*xTe';
[~, idx]    = sort(dist);
yTeHat      = mode(yTr(idx(1:k, :)), 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yTe         = yTe(:);
yTeHat      = yTeHat(:);
acc         = length(find(yTeHat == yTe))/numTe;
