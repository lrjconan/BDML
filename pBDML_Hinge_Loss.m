function [M, info] = pBDML_Hinge_Loss(X, Y, set_S, set_D, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Bounded Distortion Metric Learning with Pairwise Constraints
%
% In
%   X 			(N x D) : data matrix (one sample per row)
%   Y 			(N x 1) : label matrix
%   set_S        (dS x 2) : pair set of constraints (center, target)
%   set_D        (dT x 2) : pair set of constraints (center, imposter)
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
    dim_std         = 3*dim + 1;    % transformed dimension in standard feasibility problem
    K               = para.K;

    alpha           = para.alpha;      % margin parameter
    beta            = para.beta;      % margin parameter

    obj_upb         = para.upb;
    flagDiag        = para.flagDiag;
    num_S           = size(set_S, 1);
    num_D           = size(set_D, 1);

    if strcmp(flagDiag, 'diagonal')
        numEqns     = dim;
    else
        flagDiag    = 'full';
        numEqns     = dim*(dim+1)/2;    % we only deal with uppper triangular part due to symmetry
    end

    numCons     	= 4*numEqns + 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Construct Constraints
  C           	= zeros(dim);
  A           	= cell(numCons, 1);
  b           	= zeros(numCons, 1);

  %% construct the constraint of objective function
  idx = 1;
  bias_loss = 0;
  pair_loss               = zeros(num_S + num_D, 1);
  pair_loss(1:num_S)      = alpha - beta;
  pair_loss(1+num_S:end)  = alpha + beta;
  const_S = -(alpha - beta)/(num_S + num_D);
  const_D = -(alpha + beta)/(num_S + num_D);

  for t = 1 : num_S
    i       = set_S(t, 1);     % current point
    j       = set_S(t, 2);     % target neighbor
    diff    = X(i, :) - X(j, :);
    tmp     = diff'*diff;
    pair_loss(idx) = pair_loss(idx) + sum(sum(M.*tmp));

    if pair_loss(idx) > 0
      C         = C + tmp;
      bias_loss = bias_loss + const_S;
    end

    idx = idx + 1;
  end

  for t = 1 : num_D
    i       = set_D(t, 1);     % current point
    j       = set_D(t, 2);     % imposter neighbor
    diff    = X(i, :) - X(j, :);
    tmp     = diff'*diff;
    pair_loss(idx) = pair_loss(idx) - sum(sum(M.*tmp));

    if pair_loss(idx) > 0
      C         = C - tmp;
      bias_loss = bias_loss + const_D;
    end

    idx = idx + 1;
  end

  C     = C./(num_S + num_D);
  b(1)  = bias_loss + obj_upb;
  tmp   = zeros(dim_std);
  tmp(1:dim, 1:dim) = C;
  A{1}      = tmp;
  obj_init  = sum(sum(C.*M)) - bias_loss;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% construct the constraint from 2 to 2*numEqns + 1
    idxA        = 1;
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
            A{idxA + k} = sparse([xx xx + dim dim_std], [yy yy + dim dim_std], ...
                                 tmpVal, dim_std, dim_std);
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

            A{idxA + k} = sparse([xx xx + dim dim_std], [yy yy + dim dim_std], ...
                                 tmpVal, dim_std, dim_std);
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

            A{idxA + k} = sparse([xx xx + 2*dim dim_std], [yy yy + 2*dim dim_std], ...
                                 tmpVal, dim_std, dim_std).*const_norm;
            A{idxA + k + numEqns} = -A{idxA + k};
        end
    else
        for k = 1 : numEqns
            xx = ii(k); yy = jj(k);

            if xx == yy
                tmpVal = [1 1 -K];
                A{idxA + k} = sparse([xx xx + 2*dim dim_std], [yy yy + 2*dim dim_std], ...
                                     tmpVal, dim_std, dim_std).*const_norm;
            else
                tmpVal = [1 1 0];
                A{idxA + k} = sparse([xx xx + 2*dim dim_std], [yy yy + 2*dim dim_std], ...
                                     tmpVal, dim_std, dim_std).*const_norm;
            end

            A{idxA + k + numEqns}   = -A{idxA + k};
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% construct the m + 4*numEqns + 2 constraint
    A{numCons} = sparse(dim_std, dim_std, 1, dim_std, dim_std);
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
		obj           = sum(sum(C.*M)) - bias_loss;

		if ~para.quiet
			fprintf('Initial Objective Value = %e || Final Objective Value = %e \n', ...
				obj_init, obj);
            fprintf('pBDML: Distortion = %e \n', distortion);
		end
	else
		if ~para.quiet
			fprintf('Cost Time: %6.2f || The problem is infeasible!\n', time);
		end
	end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Return Solution
    info.obj_init            = obj_init;

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
