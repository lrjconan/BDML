function [X, w, info] = SDF_MWU_solver(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code implements the multiplicative weights update method (MWU) for 
% approximately solving the semidefinite feasibility problem below:
%
%   find    
%   s.t.    A*X     >= b
%           X       >= 0
%           Tr(X)   <= R
%
% IN
%   A : [m x 1] cell array of which element is n x n sparse matrix 
%   b : [m x 1] constraint value
%
% OUT
%   X : [n x n] optimal solution
%   obj : final objective value
%
% Reference:
%   [1] Satyen Kale. 
%   "Efficient Algorithms using the Multiplicative Weights Update Method." 
%   PhD Thesis, Princeton, 2007.
%
% Author:
%   Renjie Liao
%
% Date:
%   V_0_0 :  2014.05.02
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin < 3 || nargin > 4
        error('Incorrect number of input arguments!');
    end
	
    A           = varargin{1};
    b           = varargin{2};    
    dim         = size(A{1}, 1);
    num_cons    = numel(A);    
    
    if nargin == 3
        %% Weight Initialization       
        para    = varargin{3};
        w       = ones(num_cons, 1);
    %     w = rand(num_cons, 1);
        w       = w./(sum(w) + eps);              
    else
        %% Resume from last weight
        w       = varargin{3};
        para    = varargin{4};
    end

    if length(A) ~= length(b)
        error('The number of constraint parameters is inconsistent!');
    end

    if para.traceBound <= 0
        error('The trace bound should be greater than zero!');
    end

    if para.rho <= 0
        error('The width parameter should be greater than zero!');
    end

    if para.iterMax <= 0
        error(['The maximum iterations of multiplicative weights update' ...
            ' method should be greater than zero!']);
    end
    
    %% setting parameters       
    R               = para.traceBound;
    rho             = para.rho;             % width of the constraint
    delta           = para.delta;           % delta for ORACLE
    ell             = para.ell;
    epsilon         = para.epsilon;         % epsilon for Multiplicative Update method
    iterMax         = para.iterMax;   
    save_iter       = para.saveIter;
    disp_iter       = para.dispIter;
    save_path       = para.savePath;
    optEig.tol      = para.tolEig;
    optEig.maxit    = para.iterMaxEig;    
    optEig.issym    = 1;
    optEig.isreal   = 1;    
    
    %% preprocessing
    if ~isdir(save_path)
        mkdir(save_path);
    end    
    
    % set random seed for lanczos algorithm
    rng('shuffle');
    
    % precompute constraints
    constriants = cell(num_cons, 1);    
    
    for i = 1 : num_cons
        constriants{i} = sparse(A{i} - (b(i)/R).*speye(dim));
    end
    
    %% main iteration     
    iter        = 1;
    failOracle  = 0;
    flag_dec    = 0;
    X           = zeros(dim);
    cvT         = zeros(iterMax, 1);
    
    %% Multiplicative Weights Update method
    while iter <= iterMax        
        if mod(iter, disp_iter) == 1     
            tic;            
        end
        
        %% construct constraint matrix B (c++ version)
        B = sparse(UpdateConstraints(w, constriants, dim));
        B = B + B' - diag(diag(B));
        
        % call ORACLE : approximately compute the largest eigenvector
        [eigV, eigD, fLargeEig] = eigs(B, 1, 'LA', optEig);
        
        if fLargeEig && ~para.quiet
            warning('ORACLE (Largest Eigenvalue Problem) does not converge!');
        end
        
        tmpX = (eigV*eigV').*R;
        
        if R*eigD < -delta
            X = tmpX;
            failOracle = 1;
            break;
        end
        
        %% c++ version
        [w, cv] = UpdateWeights(w, tmpX, A, b, rho, epsilon, ell);    
        % cvT(iter+1) = sum(abs(cv));
        
        X = X + tmpX./iterMax;
%         fprintf('Current Obj Violation: %15.2f\n', cv(end));                        
        
        if ~isreal(w)
            error('The width parameter is too small!');
        end        
        
        if any(w < 0)
            error('Negative weight: increase rho!');
        end
        
        if mod(iter, disp_iter) == 0
            time    = toc;
            X_save  = X.*(iterMax/iter);
            dim_raw = (size(X_save, 1) - 1)/3;
            distortion_disp = cond(X_save(1:dim_raw, 1:dim_raw));            
            fprintf('[SDF Solver]: Iter = %05d || Distortion = %.2e || ', iter, distortion_disp); 
            fprintf('Max Pos CV = %.2e || Max Neg CV = %.2e || Time = %5.2f\n', ...
                max(cv)/rho, -min(cv)/ell, time);
            
            if distortion_disp < 1000 && ~flag_dec
                % decrease the weight
                flag_dec    = 1;
                const_norm  = 1.0e+3;
                dim         = 70;
                d2          = dim^2;
                numT        = 5288;
                numEqns     = dim*(dim+1)/2;
                idxA        = numT + 2*numEqns + 1;
                
                [ii, jj]    = ind2sub([dim dim], 1:d2);
                idxTriu     = find(ii <= jj);
                ii          = ii(idxTriu);
                jj          = jj(idxTriu);
                
                for k = 1 : numEqns
                    xx = ii(k); yy = jj(k);
                
                    if xx == yy
                        A{idxA + k} = A{idxA + k}./const_norm;
                        A{idxA + k + numEqns} = A{idxA + k}./const_norm;
                    end
                end
                
            end
            
        end
        
        if mod(iter, save_iter) == 0
            X_save  = X.*(iterMax/iter);
            
            % plot the distribution of weight
            h = figure; plot(w);
            print(h, fullfile(save_path, sprintf('weight_%07d.png', iter)), '-dpng');            
            close(h);            
            
            save(fullfile(save_path, sprintf('X_iter_%05d.mat', iter)), 'X_save', 'w');  
        end
        
        iter = iter + 1;
    end
 
    if ~failOracle        
        info.fail   = failOracle;
        info.cv     = cvT;
    else
        info.fail   = 1;
    end
        
    % Output current progress
    if ~para.quiet
        fprintf('Call % 6d times Oracle\n', iter);                   
    end
end
