% Example on the MovieLens-100k dataset. 
%
% Reference:
% 
% S. Dong, P.-A. Absil, and K. A. Gallivan. Riemannian gradient descent
% methods for graph-regularized matrix completion. Linear Algebra and its Applications, 2020. 
% DOI: https://doi.org/10.1016/j.laa.2020.06.010.
% 
% Contact: Shuyu Dong (shuyu.dong@uclouvain.be), ICTEAM, UCLouvain.
% 
% Latest version: September, 2021.
 
clear; close all;  

% Load the input data Mat from the MovieLens 100k dataset 
data = load_data('ML100k'); 
Mat = data.mat; % 


DIMS = data.dims;    
RANK_TRUE = -1;     %  A negative value indicates that an optimal rank is unknown


%% Convert the input (sparse) matrix into the unified format 'mcinfo' for GRMC 

% SR is the (optional) parameter that defines the sampling rate, when the input
% matrix X is already sparse, SR indicates the sampling rate among the known
% entries. 

SR = 0.9; 
mcinfo = GRMC.build_mcinfo_from_sparse_matrix(Mat, SR); 


%% Generate graph laplacian matrices 


feat =  gen_features_for_graphinfo(mcinfo, 'rows'); 
[ginfo_r, t] = gen_graphlaplacian_from_features(feat); %  
Lr = ginfo_r.L; % 


% In this example, we are happy with just a row-wise similarity graph and set
% the column-wise graph to zero. If one needs a non-zero graph on column
% indices, just use  
%   feat =  gen_features_for_graphinfo(mcinfo, 'cols') 
% to get the column-wise features before generating the graph Laplacian 

Lc = sparse(DIMS(2)); 
 


%% Set up regularization parameters and then construct the GRMC problem object
% parameter (1):    alpha_r
% parameter (2):    alpha_c
% parameter (3):    Lreg_gammar
% parameter (4):    Lreg_gammac 
% parameter (5):    delta (parameter in the preconditioned metric): 
%                       Suggested value is 1e-8. A smaller value induces 
%                       a Riemannian gradient that is closer to the Newton 
%                       direction. 

PARAMS = [0.39, 0.39, 0.2, 0.2, 1e-8]; 
RANK = 12; 


pb = GRMC(mcinfo, RANK, PARAMS, Lr, Lc); 


%% Set up solver parameters and initialize solver 

MAXTIME = 20;
MAXIT = 5000; 
TOL = 1e-12;
VERBO = 1; 

so = Solver(pb, 'maxtime', MAXTIME, 'maxiter', MAXIT, ...
                'tolgradnorm', TOL, 'verbosity', VERBO); 

%% Solve the problem 
% Generate initial point 

Xinit = pb.initialization('M0', RANK); 


% Choose one solver 
for so_i = 1 : 3 
     
    % Run solver 
    
    switch so_i 
        case 1 
            [X_, stats_]  = so.solve_RGD_lsRBB2(Xinit); 
            m_name  = 'Precon RGD (RBB)'; 
        case 2 
            [X_, stats_]  = so.solve_RCG_lsFree(Xinit);
            m_name  = 'Precon RCG (linemin)'; 
        case 3
            [X_, stats_]  = so.solve_RGD_lsFree(Xinit);
            m_name  = 'Precon RGD (linemin)'; 
    end
    
    res{so_i} = stats_; 
    m_names{so_i} = m_name; 
end


%% Show figure and table 
cc = parula(3);  % 

h(1) = figure(); 
for i = 1 : 3
    semilogx([res{i}.time],[res{i}.RMSE_tr], 'Marker','none', 'Color', cc(i,:), 'linewidth', 2);
    hold on; 
end
xlabel('Time (seconds)'); 
ylabel('RMSE (train)'); 
legend(m_names); 


h(2) = figure(); 
for i = 1 : 3
    semilogx([res{i}.time],[res{i}.RMSE_t], 'Marker','none', 'Color', cc(i,:), 'linewidth', 2);
    hold on; 
end
xlabel('Time (seconds)'); 
ylabel('RMSE (test)'); 
legend(m_names); 


h(3) = figure(); 
for i = 1 : 3
    semilogy([res{i}.time],[res{i}.gradnorm], 'Marker','none', 'Color', cc(i,:), 'linewidth', 2);
    hold on; 
end
xlabel('Time (seconds)'); 
ylabel('gradnorm'); 
legend(m_names); 

%% Show performances of the solution 

for i = 1 : 3
    temp = struct(); 
    temp = res{i}(end) ; 
    temp.Alg = m_names{i}; 
    tab_res(i,:) = struct2table(temp, 'AsArray',true); 
end

disp(tab_res(:,{'Alg', 'RMSE_tr','RMSE_t'})); 

% Results from one random test are as follows
%   
%               Alg               RMSE_tr    RMSE_t
%     ________________________    _______    _______
% 
%     {'Precon RGD (RBB)'    }    0.76267    0.95098
%     {'Precon RCG (linemin)'}    0.76267    0.95098
%     {'Precon RGD (linemin)'}     0.7633    0.95046



