
clear; 

% The input data Mat is a MATLAB (dense or sparse) matrix 

DIMS = [100, 200]; 
RANK_TRUE = 12; 

Mat = randn(DIMS(1), RANK_TRUE)*randn(RANK_TRUE, DIMS(2)); 

%% Convert input data into the unified input format 'mcinfo' for GRMC 

% SR is the (optional) parameter that defines the sampling rate, when the input
% matrix X is already sparse, SR indicates the sampling rate among the known
% entries. 

SR = 0.3; 
mcinfo = GRMC.build_mcinfo_from_sparse_matrix(Mat, SR); 

%% Set up regularization parameters and then construct the GRMC problem object
% parameter (1):    alpha_r
% parameter (2):    alpha_c
% parameter (3):    Lreg_gammar
% parameter (4):    Lreg_gammac 
% parameter (5):    delta (parameter in the preconditioned metric): 
%                       Suggested value is 1e-8. A smaller value induces 
%                       a Riemannian gradient that is closer to the Newton 
%                       direction. 

PARAMS = [0.0, 0.0, 0, 0, 1e-8]; 
RANK = 12; 

Lr = sparse(DIMS(1)); 
Lc = sparse(DIMS(2)); 

pb = GRMC(mcinfo, RANK, PARAMS, Lr, Lc); 


%% Set up solver parameters and initialize solver 

MAXTIME = 100;
MAXIT = 5000; 
TOL = 1e-12;
VERBO = 1; 

so = Solver(pb, 'maxtime', MAXTIME, 'maxiter', MAXIT, ...
                'tolgradnorm', TOL, 'verbosity', VERBO); 

%% Solve the problem 
% Generate initial point 

Xinit = pb.initialization('M0', RANK); 

% Choose and run solver 
so_i = 1 ; 

switch so_i 
    case 1 
        [X_, stats_]  = so.solve_RGD_lsRBB2(Xinit); 
    case 2 
        [X_, stats_]  = so.solve_RCG_lsFree(Xinit);
    case 3
        [X_, stats_]  = so.solve_RGD_lsFree(Xinit);
    % case 4 
    %     [X_, stats_]  = so.solve_rsd_manopt(Xinit);
    % case 5
    %     [X_, stats_]  = so.solve_rcg_manopt(Xinit);
end



