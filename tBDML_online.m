function [M, info] = tBDML_online(X, Y, setS, setT, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Bounded Distortion Metric Learning for Triplet Constraints 
%
% In
%   X 			(N x D)     : data matrix (one sample per row)
%   Y 			(N x 1)     : label matrix
%   setS        (dS x 3)    : pair set of constraints (center, target)
%   setT        (dT x 3)    : triplet set of constraints (center, target, imposter)
%   para                    : parameter class
%                    para.K : bound of distortion
%
% Out
%   M (D x D) : metric matrix
%   info      : information of optimization process
%
% Author:
%   Renjie Liao
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	%% Check Input and Output
    if nargin ~= 5
        error('Incorrect number of input arguments!');
    else
        if size(X, 1) ~= size(Y, 1)
            error('The numbers of data and labels are inconsistent!');
        end               
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
    %% Setting Parameters
    [num, dim]      = size(X);
    d2              = dim^2;
    dimStd          = 3*dim + 1;    % transformed dimension in standard feasibility problem        
    mu              = para.mu;      % margin parameter
    objVal          = para.upb;
    K               = para.K;	
	batch_size 		= para.batchSize;
	round_num 		= para.round;	
    numS            = size(setS, 1);
    numT            = size(setT, 1);
    numEqns         = dim*(dim+1)/2;    % we only deal with uppper triangular part due to symmetry

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    round_idx = 0;
    max_num_batch = ceil(numT/batch_size);
    M_batch = zeros(dim); 
              
    %% construct the constraint of objective function
    C = zeros(dim);
    
    for t = 1 : numS                
        i       = setS(t, 1);     % current point
        j       = setS(t, 2);     % target neighbor             
        diffS   = X(i, :) - X(j, :);          
        C       = C + diffS'*diffS;    % Rank-1 update
    end

    C = C./numS; 
    const_norm = 1/(norm(C, 'fro') + eps);
    initObj = trace(C);     % initial metric is set to Euclidean    
    
    while round_idx < round_num
        batch_idx = 0;
        
        while batch_idx < max_num_batch        
            idx_setT = (batch_idx * batch_size + 1) : min(numT, (batch_idx+1) * batch_size);            
            setT_local_copy = setT(idx_setT, :);
            numT_local_copy = size(setT_local_copy, 1);
            
            %% Construct Constraints                                    
            numCons = numT_local_copy + 4*numEqns + 2;
            
            A                   = cell(numCons, 1);   
            b                   = zeros(numCons, 1);
            tmp                 = zeros(dimStd);
            tmp(1:dim, 1:dim)   = triu(-C.*const_norm);
            A{1}                = sparse(tmp);
            b(1)                = -objVal*const_norm;      
            
            %% construct margin constraints from 2 to m+1
            idxA    = 1;
            tmp     = zeros(dimStd);

            for t = 1 : numT_local_copy                
                i                   = setT_local_copy(t, 1);     % current point
                j                   = setT_local_copy(t, 2);     % target neighbor
                k                   = setT_local_copy(t, 3);     % imposter neighbor        
                diffS               = X(i, :) - X(j, :);
                diffI               = X(i, :) - X(k, :);
                xS                  = diffS'*diffS;
                xI                  = diffI'*diffI;
                tmpA                = xI - xS;        
                tmp(1:dim, 1:dim)   = triu(tmpA);
                A{idxA + t}         = sparse(tmp).*const_norm;      % save upper triangular part due to symmetry
                b(idxA + t)         = mu*const_norm;
            end

            %% construct the constraint from m+2 to m + 2*numEqns + 1
            idxA        = idxA + numT_local_copy;
            [ii, jj]    = ind2sub([dim dim], 1:d2);
            idxTriu     = find(ii <= jj);
            ii          = ii(idxTriu);
            jj          = jj(idxTriu);

            for k = 1 : numEqns
                xx = ii(k); yy = jj(k);

                if xx == yy
                    tmpVal = [1 -1 -1];
                else
                    tmpVal = [1 -1 0];
                end

                A{idxA + k} = sparse([xx xx + dim dimStd], [yy yy + dim dimStd], ...
                                     tmpVal, dimStd, dimStd);                
                A{idxA + k + numEqns} = -A{idxA + k};
            end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% construct the constraint from m + 2*numEqns + 2 to m + 4*numEqns + 1
            idxA = idxA + 2*numEqns;

            for k = 1 : numEqns
                xx = ii(k); yy = jj(k);

                if xx == yy
                    tmpVal      = [1 1 -K];
                    A{idxA + k} = sparse([xx xx + 2*dim dimStd], [yy yy + 2*dim dimStd], ...
                                         tmpVal, dimStd, dimStd).*para.lambda_distortion;            
                else
                    tmpVal      = [1 1 0];
                    A{idxA + k} = sparse([xx xx + 2*dim dimStd], [yy yy + 2*dim dimStd], ...
                                         tmpVal, dimStd, dimStd).*para.lambda_distortion;             
                end

                A{idxA + k + numEqns}   = -A{idxA + k};
            end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            %% construct the m + 4*numEqns + 2 constraint
            A{numCons} = sparse(dimStd, dimStd, 1, dimStd, dimStd);
            b(numCons) = eps;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            %% Solve Convex Feasibility Problem via the MWU solver    
            tic;
            [hatM, w, infoMWU] = SDF_MWU_solver(A, b, para.solverMWU);
            time = toc;

            %% update the objective value
            if ~infoMWU.fail
                M             = hatM(1:dim, 1:dim);
                distortion    = cond(M);
                obj           = sum(sum(C.*M));

                if ~para.quiet
                    fprintf('Current & Initial Objective Value: %8.7e & %8.7e || Cost Time: %6.2f\n', ...
                        obj, initObj, time);
                end
            else
                if ~para.quiet
                    fprintf('Cost Time: %6.2f || The problem is infeasible!\n', time);
                end
            end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Return Solution    
            info.initObj            = initObj;

            if ~infoMWU.fail
                info.obj            = obj;       
                info.distortion     = distortion;
                info.w              = w;
                info.MWU_solver     = infoMWU;
                info.fail           = 0;
            else
                M                   = zeros(dim);
                info.fail           = 1;
            end

            M_batch = M_batch + M./max_num_batch;
            batch_idx = batch_idx + 1;
        end
        
        round_idx = round_idx + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%