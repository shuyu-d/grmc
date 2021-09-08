function [x, info, rperi] = solve_RGD_lsBB1(self, method_init, varargin)
% This function implements the Riemannian Steepest Descent algorithm.

if nargin > 2 
% Update optimization-related options from varargin
self.params_opt = self.set_params(varargin{:}); % Solver.parseArgs(varargin{:});
end 

% Verify that the problem description is sufficient for the solver.
if ~isa(self.problem.grad, 'function_handle') %~canGetGradient(problem) && ~canGetApproxGradient(problem)
    % Note: we do not give a warning if an approximate gradient is
    % explicitly given in the problem description, as in that case the
    % user seems to be aware of the issue.
    warning('Solver: ', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
end
% Set local defaults here.
% localdefaults.minstepsize = 1e-10;
localdefaults.maxiter = 1000;
localdefaults.tolgradnorm = 1e-6;
% Merge user-defined options into local defaults (the 2nd input arg.
% overrides the 1st one), see manopt/core/mergeOptions.m.  
% When this function is called `paramsopt is     
% already set via Tester.parseArgs('opt',varargin{:})  in Tester() or run_tester().
options = mergeOptions(localdefaults, self.params_opt); 
options.stopfun = self.problem.stopfun; 

% The step size selection is done by line-search with an initial 
% stepsize estimation. All Solver.stepsize_init_*.m functions have a
% unified input arguments list (x, d, itersdb, store, manifold). 
switch options.stepsize0_type
    case 'exact'
        % This is ready to use
        options.stepsize0_fun = @(x,d, itersdb, store)...
                       self.problem.stepsz_estimator(x,d,store); 
    case 'lipschitz'
        % This is ready to use
        options.stepsize0_fun = @(x,d, itersdb, store)...
                       Solver.stepsize_init_lipschitz(x,d,itersdb, store, self.problem.manifold); 
    case 'BB'
        % (todo) This is to be implemented 
        options.stepsize0_fun = @(x,d, itersdb,store)...
                       Solver.stepsize_init_BB(x,d,itersdb,store, self.problem.manifold); 
    case 'one'
        % This is ready to use
        options.stepsize0_fun = @(x,d,itersdb,store) .5; 
end
% 
% % note (#501): here we assume that the line search method is not none. To
% % implement a RGD algorithm *without line search*, write a separate function
% % with the name in @Solver/slove_RGD_lsfree.m 
% % note (#511): to improve the efficiency of Solver, drop the option of
% % choosing the linesearch method inside solve_RCG.m, instead, write in
% % different functions for each linesearch method, such as
% % @Solver/solve_{RCG_lsArmijo, RCG_lsfree, RCG_lsNonmonotone}.m 
% % todo: rename this function to solve_RGD_lsArmijo.m 

DEPTH = 2;
it_cyc = @(iter) mod(iter, DEPTH)+1 ;

% Create a database to caching the iteration-related information:
% the depth of this database is limited to 2, for the current and the new
% iterates. By convention, the index for the current iterate 
% is given by it_cyc(iter). 
itersdb(1) = struct('iter',nan, 'rgrad',[],'fobj',nan, 'x',...
               [], 'cache', struct(), 'stepsize0',[]);
itersdb(2) = struct('iter',nan, 'rgrad',[],'fobj',nan, 'x', ...
               [], 'cache', struct(), 'stepsize0',[]);

timetic = tic();

% Initialization: 
x = self.initialization(method_init); 

% Iteration counter.
% At any point, iter is the number of fully executed iterations so far.
iter = 0;
    
% Compute (cost,grad) quantities for x. 
% [cost, itersdb(it_cyc(iter)).cache] = self.cost(x, itersdb(it_cyc(iter)).cache ); 
cost = nan; 
[grad, itersdb(it_cyc(iter)).cache] = self.problem.grad(x, itersdb(it_cyc(iter)).cache);

% At the same time fobj,rgrad, x is stored in itersdb(iter_current):
itersdb(it_cyc(iter)).rgrad = grad; 
itersdb(it_cyc(iter)).fobj = cost; 
itersdb(it_cyc(iter)).x = x; 

gradnorm = self.problem.manifold.norm(x, grad);
% here we have (x, cost, grad, itersdb) ready for iter = 0. 
% Save stats in a struct array info, and preallocate.
stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];
stepsize = nan; 
if options.verbosity >= 2
    fprintf(' iter\t    stepsize\t    grad. norm    RMSE(tr) \n');
end

% Start iterating until stopping criterion triggers.
desc_dir = self.problem.manifold.lincomb(x, -1, grad);
stepsize = options.stepsize0_fun(x, desc_dir, itersdb, ...
                                      itersdb(it_cyc(iter-1)).cache) ; 
%stepsize = 1.e-2;
while true
    gradk = grad; xk = x;
    
    % Display iteration information.
    if options.verbosity >= 2
        fprintf('%5d\t%+.2e\t%+.4e\t%.4e\n', iter, stepsize, gradnorm, stats.RMSE_tr);
    elseif options.verbosity == 1 && ~mod(iter, 50)
        fprintf('%5d\t%+.2e\t%+.4e\t%.4e\n', iter, stepsize, gradnorm, stats.RMSE_tr);
    end
    
    % Start timing this iteration.
    timetic = tic();
    
    % Run standard stopping criterion checks.
    [stop, reason] = stoppingcriterion(self, xk, options, info, iter+1);
    
    % If none triggered, run specific stopping criterion check.
    if ~stop && stats.stepsize < options.minstepsize
        stop = true;
        reason = sprintf(['Last stepsize smaller than minimum '  ...
                          'allowed; options.minstepsize = %g.'], ...
                          options.minstepsize);
    end

    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end

    % Pick the descent direction as minus the gradient, based on info from
    % (x, cost, grad, itersdb) of the iteration #iter. 
    desc_dir = self.problem.manifold.lincomb(xk, -1, gradk);
    
    % The initial stepsize is to be computed via options.stepsize0_fun(). 
%     stepsize = options.stepsize0_fun(x, desc_dir, itersdb, ...
%                                       itersdb(it_cyc(iter-1)).cache) ; 
    % Make the chosen step and compute the cost there.
    
    x = self.problem.manifold.retr(xk, desc_dir, stepsize);
    
    % Release cache in itersdb for the new (next) iter
    itersdb(it_cyc(iter)).cache = struct(); 
    % Compute the new grad-related quantities for x
    % [cost, itersdb(it_cyc(iter)).cache] = self.cost(x, itersdb(it_cyc(iter)).cache ); 
    [grad, itersdb(it_cyc(iter)).cache] = self.problem.grad(x, itersdb(it_cyc(iter)).cache ); 

    % At the same time fobj,rgrad, x is stored in itersdb(iter_current):
    itersdb(it_cyc(iter)).rgrad = grad; 
    itersdb(it_cyc(iter)).fobj = cost; 
    itersdb(it_cyc(iter)).x = x; 
    gradnorm = self.problem.manifold.norm(x, grad);
    % here we have (x, cost, grad, itersdb) ready for #iter = iter+1. 
    
    % ---------------- Bin add ----------------------
%     x = xk + stepsize*desc_dir;
   
    % BB
    Y = self.problem.manifold.lincomb(x, 1, grad, -1, gradk);
    S = self.problem.manifold.lincomb(x, 1, x, -1, xk);  
    SY = abs(self.problem.manifold.inner(x,S,Y));
    SS = self.problem.manifold.norm(x,S)^2;
%     YY = self.problem.manifold.norm(x,Y)^2;
%     if mod(iter,2)==0
        stepsize = SS/SY;
%     else
%         stepsize  = SY/YY;
%     end
    stepsize = max(min(stepsize, 1e15), 1e-15);
    % ------------------------------------------------
    
    % iter is the number of iterations we have accomplished.
    iter = iter + 1;

    % Log statistics for freshly executed iteration.
    stats = savestats();
    info(iter+1) = stats;
end


info = info(1:iter+1);

if options.verbosity >= 1
    fprintf('Total time is %f [s] (excludes statsfun)\n', ...
            info(end).time);
end

% % Store results per-iteration (rperi) in a class-standard struct.
% rperi = self.get_rperi(info, 'RGD_lsFree'); 

% Routine in charge of collecting the current iteration stats
function stats = savestats()
    stats.iter = iter;
    stats.cost = cost;
    stats.gradnorm = gradnorm;
    if iter == 0
        stats.stepsize = NaN;
        stats.time = toc(timetic);
        stats.linesearch = [];
    else
        stats.stepsize = stepsize;
        stats.time = info(iter).time + toc(timetic);
        stats.linesearch = [];
    end
    stats = self.problem.statsfun([], x, stats, itersdb(it_cyc(iter)).cache) ; 
end

end


