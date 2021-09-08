function [x, info, rperi] = solve_RCG_lsFree(self, method_init, varargin)
% This function implements the Riemannian Conjugate Gradient algorithm.

if nargin > 2 
% Update optimization-related options from varargin
self.params_opt = self.set_params(varargin{:}); % Solver.parseArgs(varargin{:});
end 

% Verify that the problem description is sufficient for the solver.
if ~isa(self.problem.cost, 'function_handle') 
    warning('Solver: ', ...
            'No cost provided. The algorithm will likely abort.');
end
if ~isa(self.problem.grad, 'function_handle') 
    % Note: we do not give a warning if an approximate gradient is
    % explicitly given in the problem description, as in that case the
    % user seems to be aware of the issue.
    warning('Solver: ', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
end

% Set local defaults here.
localdefaults.maxiter = 1000;
localdefaults.tolgradnorm = 1e-6;
% Changed by NB : H-S has the "auto restart" property.
% See Hager-Zhang 2005/2006 survey about CG methods.
% The auto restart comes from the 'max(0, ...)', not so much from the
% reason stated in Hager-Zhang I think. P-R also has auto restart.
localdefaults.beta_type = 'H-S';
localdefaults.orth_value = Inf; % by BM as suggested in Nocedal and Wright

% Merge user-defined options into local defaults (the 2nd input arg.
% overrides the 1st one), see manopt/core/mergeOptions.m.  
% When this function is called paramsopt is already set via 
% Tester.parseArgs('opt',varargin{:})  
options = mergeOptions(localdefaults, self.params_opt); 
options.stopfun = self.problem.stopfun; 

% The step size selection is done by line-search with an initial 
% stepsize estimation. All Solver.stepsize_init_*.m functions have a
% unified input arguments list (x, d, itersdb, store, manifold). 
switch options.stepsize0_type
    case Solver.NAMES_SS0{1} % 'exact' (now named linemin)
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
% [cost, itersdb(it_cyc(iter)).cache] = self.problem.cost(x, itersdb(it_cyc(iter)).cache ); 
cost = nan; 
[grad, itersdb(it_cyc(iter)).cache] = self.problem.grad(x, itersdb(it_cyc(iter)).cache ); 

% At the same time fobj,rgrad, x is stored in itersdb(iter_current):
itersdb(it_cyc(iter)).rgrad = grad; 
itersdb(it_cyc(iter)).fobj = cost; 
itersdb(it_cyc(iter)).x = x; 
itersdb(it_cyc(iter+1)) = itersdb(it_cyc(iter)) ; % initialize 
gradnorm = self.problem.manifold.norm(x, grad);

% Pgrad (in cg-manopt) is always identified with grad here since no preconditioner is applied to grad. 
gradPgrad = gradnorm^2;

% here we have (x, cost, grad, itersdb) ready for iter = 0. 
% Save stats in a struct array info, and preallocate.
stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];
stepsize = nan; 
if options.verbosity >= 1
    fprintf(' iter\t     stepsize\t    grad. norm     RMSE(tr)\n');
end
     
% Compute a first descent direction (not normalized)
desc_dir = self.problem.manifold.lincomb(x, -1, grad);

% Start iterating until stopping criterion triggers.
while true

    % Display iteration information.
    if options.verbosity >= 2
        fprintf('%5d\t%+.2e\t%+.4e\t%.4e\n', iter, stepsize, gradnorm,...
                stats.RMSE_tr);
    elseif options.verbosity == 1 && ~mod(iter, 50)
        fprintf('%5d\t%+.2e\t%+.4e\t%.4e\n', iter, stepsize, gradnorm, stats.RMSE_tr);
    end
    
    % Start timing this iteration.
    timetic = tic();
    
    % Run standard stopping criterion checks.
    [stop, reason] = stoppingcriterion(self, x, options, info, iter+1);
    
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

    % The line search algorithms require the directional derivative of the
    % cost at the current point x along the search direction.
    df0 = self.problem.manifold.inner(x, grad, desc_dir);
        
    % If we didn't get a descent direction: restart, i.e., switch to the
    % negative gradient. Equivalent to resetting the CG direction to a
    % steepest descent step, which discards the past information.
    if df0 >= 0
        if options.verbosity >= 3
            fprintf(['Conjugate gradient info: got an ascent direction '...
                     '(df0 = %2e), reset to the (preconditioned) '...
                     'steepest descent direction.\n'], df0);
        end
        % Reset to negative gradient: this discards the CG memory.
        desc_dir = self.problem.manifold.lincomb(x, -1, grad);
        df0 = -gradPgrad;
    end

    % Compute the step-size. 
    stepsize = options.stepsize0_fun(x, desc_dir, itersdb, ...
                                      itersdb(it_cyc(iter-1)).cache) ; 
    % Make the CG step with the stepsize. 
    newx = self.problem.manifold.retr(x, desc_dir, stepsize);
    
    % Release cache in itersdb for the new (next) iter
    itersdb(it_cyc(iter)).cache = struct(); 
    
    % Compute the new grad-related quantities for x
    [newgrad, itersdb(it_cyc(iter)).cache] = self.problem.grad(newx, itersdb(it_cyc(iter)).cache ); 
    % At the same time fobj,rgrad, x is stored in itersdb(iter_current):
    itersdb(it_cyc(iter)).rgrad = newgrad; 
    itersdb(it_cyc(iter)).x = newx; 

    % Compute the gradnorm of the new gradient. 
    newgradnorm = self.problem.manifold.norm(newx, newgrad);
    % Recall that Pnewgrad = newgrad. 
    newgradPnewgrad = newgradnorm^2; 

    % Apply the CG scheme to compute the next search direction.
    % This paper https://www.math.lsu.edu/~hozhang/papers/cgsurvey.pdf
 	% by Hager and Zhang lists many known beta rules. The rules defined
    % here can be found in that paper (or are provided with additional
    % references), adapted to the Riemannian setting.
 	 
    oldgrad = self.problem.manifold.transp(x, newx, grad);
    orth_grads = self.problem.manifold.inner(newx, oldgrad, newgrad) / newgradPnewgrad;
    
    % Powell's restart strategy (see page 12 of Hager and Zhang's
    % survey on conjugate gradient methods, for example)
    if abs(orth_grads) >= options.orth_value
        beta = 0;
        desc_dir = self.problem.manifold.lincomb(x, -1, newgrad);
        
    else % Compute the CG modification
        desc_dir = self.problem.manifold.transp(x, newx, desc_dir);
        switch upper(options.beta_type)
            case 'F-R'  % Fletcher-Reeves
                beta = newgradPnewgrad / gradPgrad;
            
            case 'P-R'  % Polak-Ribiere+
                % vector grad(new) - transported grad(current)
                diff = self.problem.manifold.lincomb(newx, 1, newgrad, -1, oldgrad);
                ip_diff = self.problem.manifold.inner(newx, newgrad, diff);
                beta = ip_diff / gradPgrad;
                beta = max(0, beta);
            
            case 'H-S'  % Hestenes-Stiefel+
                diff = self.problem.manifold.lincomb(newx, 1, newgrad, -1, oldgrad);
                ip_diff = self.problem.manifold.inner(newx, newgrad, diff);
                beta = ip_diff / self.problem.manifold.inner(newx, diff, desc_dir);
                beta = max(0, beta);
 
            case 'H-Z' % Hager-Zhang+
                diff = self.problem.manifold.lincomb(newx, 1, newgrad, -1, oldgrad);
                % Poldgrad = self.problem.manifold.transp(x, newx, grad);//==oldgrad
                % Pdiff = self.problem.manifold.lincomb(newx, 1, newgrad, -1, oldgrad);//==diff
                deno = self.problem.manifold.inner(newx, diff, desc_dir);
                numo = self.problem.manifold.inner(newx, diff, newgrad);
                numo = numo - 2*self.problem.manifold.inner(newx, diff, diff)*...
                              self.problem.manifold.inner(newx, desc_dir, newgrad) / deno;
                beta = numo / deno;
 
                % Robustness (see Hager-Zhang paper mentioned above)
                desc_dir_norm = self.problem.manifold.norm(newx, desc_dir);
                eta_HZ = -1 / ( desc_dir_norm * min(0.01, gradnorm) );
                beta = max(beta, eta_HZ);
            otherwise
                error(['Unknown options.beta_type. ' ...
                       'Should be steep, S-D, F-R, P-R, H-S or H-Z.']);
        end
        
        desc_dir = self.problem.manifold.lincomb(newx, -1, newgrad, beta, desc_dir);
    end
    
    % Transfer iterate info.
    x = newx;
    grad = newgrad;
    gradnorm = newgradnorm;
    gradPgrad = newgradPnewgrad;
    
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
% rperi = self.get_rperi(info, 'RCG_lsFree'); 

     % Routine in charge of collecting the current iteration stats
     function stats = savestats()
         stats.iter = iter;
         stats.cost = NaN;
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

