function [M, info] = pBDML(X, Y, setS, setD, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Bounded Distortion Metric Learning with Pairwise Constraints 
%
% In
%   X 			(N x D) : data matrix (one sample per row)
%   Y 			(N x 1) : label matrix
%   setS        (dS x 3) : pair set of constraints (center, target)
%   setD        (dT x 3) : pair set of constraints (center, imposter)
%   para      			: parameter class
%   para.K : bound of distortion        
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
    K               = para.K;
    mu              = para.mu;      % margin parameter
    objVal          = para.upb;    
    flagDiag        = para.flagDiag;    
    numS            = size(setS, 1);
    numD            = size(setD, 1);
    
    if strcmp(flagDiag, 'diagonal')        
        numEqns     = dim;
    else
        flagDiag    = 'full';
        numEqns     = dim*(dim+1)/2;    % we only deal with uppper triangular part due to symmetry
    end      
           
    numCons     	= numD + 4*numEqns + 2;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Construct Constraints
    C           	= zeros(dim);
    A           	= cell(numCons, 1);   
    b           	= zeros(numCons, 1);    
    
    %% construct the constraint of objective function
	for t = 1 : numS                
        i       = setS(t, 1);     % current point
        j       = setS(t, 2);     % target neighbor             
        diffS   = X(i, :) - X(j, :);          
        C       = C + diffS'*diffS;    % Rank-1 update
	end
    
    C                   = C./numS;
    tmp                 = zeros(dimStd);
    
    if strcmp(flagDiag, 'diagonal')
        tmp(1:dim, 1:dim)   = diag(diag(-C));
    else
        tmp(1:dim, 1:dim)   = triu(-C);
    end
    
    A{1}                = sparse(tmp);
    b(1)                = -objVal;    
    const_norm          = norm(A{1}, 'fro') + eps;        
	A{1}                = A{1}./const_norm;
    b(1)                = b(1)/const_norm;
	initObj             = trace(C);     % initial metric is set to Euclidean
        
    %% construct pair constraints from 2 to m+1
    idxA    = 1;
    tmp     = zeros(dimStd);
	
    if strcmp(flagDiag, 'diagonal')
        for t = 1 : numD
            i                   = setD(t, 1);     % current point
            j                   = setD(t, 2);     % imposter neighbor        
            diffD               = X(i, :) - X(j, :);
            xI                  = diffD'*diffD;
            tmp(1:dim, 1:dim)   = diag(diag(xI));            
            A{idxA + t}         = sparse(tmp)./const_norm;      % save upper triangular part due to symmetry
            b(idxA + t)         = mu./const_norm;
        end
    else        
        for t = 1 : numD                
            i                   = setD(t, 1);     % current point
            j                   = setD(t, 2);     % imposter neighbor        
            diffD               = X(i, :) - X(j, :);
            xI                  = diffD'*diffD;
            tmp(1:dim, 1:dim)   = triu(xI);                    
            A{idxA + t}         = sparse(tmp)./const_norm;      % save upper triangular part due to symmetry
            b(idxA + t)         = mu./const_norm;
        end    
    end
	      
    %% construct the constraint from m+2 to m + 2*numEqns + 1
    idxA        = idxA + numD;
    [ii, jj]    = ind2sub([dim dim], 1:d2);
    
    if strcmp(flagDiag, 'diagonal')
        idxDiag     = find(ii == jj);
        ii          = ii(idxDiag);
        jj          = jj(idxDiag);    
    else
        idxTriu     = find(ii <= jj);
        ii          = ii(idxTriu);
        jj          = jj(idxTriu);
    end
    
    if strcmp(flagDiag, 'diagonal')
        for k = 1 : numEqns
            xx = ii(k); yy = jj(k);
            
            tmpVal = [1 -1 -1];               
            A{idxA + k} = sparse([xx xx + dim dimStd], [yy yy + dim dimStd], ...
                                 tmpVal, dimStd, dimStd);        
            A{idxA + k + numEqns} = -A{idxA + k};
        end
    else
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
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% construct the constraint from m + 2*numEqns + 2 to m + 4*numEqns + 1
    idxA = idxA + 2*numEqns;
    % weight the importantce of bounded-distortion constraint
	const_norm  = para.lambda_distortion;   
    
    if strcmp(flagDiag, 'diagonal')
         for k = 1 : numEqns
            xx = ii(k); yy = jj(k);

            tmpVal = [1 1 -K];            

            A{idxA + k} = sparse([xx xx + 2*dim dimStd], [yy yy + 2*dim dimStd], ...
                                 tmpVal, dimStd, dimStd).*const_norm;
            A{idxA + k + numEqns} = -A{idxA + k};                
        end   
    else
        for k = 1 : numEqns
            xx = ii(k); yy = jj(k);

            if xx == yy
                tmpVal = [1 1 -K];
                A{idxA + k} = sparse([xx xx + 2*dim dimStd], [yy yy + 2*dim dimStd], ...
                                     tmpVal, dimStd, dimStd).*const_norm;
            else
                tmpVal = [1 1 0];
                A{idxA + k} = sparse([xx xx + 2*dim dimStd], [yy yy + 2*dim dimStd], ...
                                     tmpVal, dimStd, dimStd).*const_norm;
            end
            
            A{idxA + k + numEqns}   = -A{idxA + k};
        end
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
        if strcmp(flagDiag, 'diagonal')
            M = diag(diag(hatM(1:dim, 1:dim)));
        else
            M = hatM(1:dim, 1:dim);
        end
            
		distortion    = cond(M);
		obj           = sum(sum(C.*M));
		
		if ~para.quiet
			fprintf('Initial Objective Value = %e || Final Objective Value = %e \n', ...
				initObj, obj);
            fprintf('pBDML: Distortion = %e \n', distortion);            
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%