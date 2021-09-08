function [X, stats, self] = solve_GD_BBstep(self, method_init, varargin)
% Recall the properties of this class:
    %properties
    %manifold;
    %cost;
    %grad;
    %hess;
    %lsstepsize_initialguess; 
    %statsfun;
    %stopfun;
    %preconditioner;
    %params_opt;
    %output;
    %scores;
    %optlog;
    %end
% quantities to maintain at each iteration: X, Xold, grad,
% grad_old
    method_init = self.params_opt.method_init;
    methodname = 'GD_BBstep';

    X0 = self.initialization(method_init);
    self.optlog.X0 = X0;
    % -------------- 1. parameter setting -----------------
    % Set local defaults here
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    options = mergeOptions(localdefaults, self.params_opt);
    
    timetic = tic();            
    % ----------- GD iterations ---------------
    %% todo: 
    %% a. change the way `storedb` is maintained, while keep its
    %%    usage in `solver.statsfun(X,stats,store)` unchanged. 
    %%    a1. rename `storedb` to `itersdb`, which will maintain
    %%    all matrices needed for the computation of cost, grad at
    %%    the current iteration(iter+1), and also `X_old, grad_old`
    %%    (iter), before writting (X,grad,..) with (iter+1). 
    %% b. 
    %% Compute objective-related quantities for x
    %[cost, grad] = getCostGrad(problem, x, storedb, key);
    %gradnorm = problem.M.norm(x, grad);
    %
    %% Iteration counter.
    %% At any point, iter is the number of fully executed iterations so far.
    %iter = 0;
    %
    %% Save stats in a struct array info, and preallocate.
    %stats = savestats();
    %info(1) = stats;
    %info(min(10000, options.maxiter+1)).iter = [];
    %
    %if options.verbosity >= 2
    %    fprintf(' iter\t               cost val\t    grad. norm\n');
    %end
    %
    %% Start iterating until stopping criterion triggers
    %while true

    %    % Display iteration information
    %    if options.verbosity >= 2
    %        fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
    %    end
    %    
    %    % Start timing this iteration
    %    timetic = tic();
    %    
    %    % Run standard stopping criterion checks
    %    [stop, reason] = stoppingcriterion(problem, x, options, ...
    %                                                         info, iter+1);
    %    
    %    % If none triggered, run specific stopping criterion check
    %    if ~stop && stats.stepsize < options.minstepsize
    %        stop = true;
    %        reason = sprintf(['Last stepsize smaller than minimum '  ...
    %                          'allowed; options.minstepsize = %g.'], ...
    %                          options.minstepsize);
    %    end
    %
    %    if stop
    %        if options.verbosity >= 1
    %            fprintf([reason '\n']);
    %        end
    %        break;
    %    end

    %    % Pick the descent direction as minus the gradient
    %    desc_dir = problem.M.lincomb(x, -1, grad);
    %    
    %    % Execute the line search
    %    [stepsize, newx, newkey, lsstats] = options.linesearch( ...
    %                         problem, x, desc_dir, cost, -gradnorm^2, ...
    %                         options, storedb, key);
    %    
    %    % Compute the new cost-related quantities for x
    %    [newcost, newgrad] = getCostGrad(problem, newx, storedb, newkey);
    %    newgradnorm = problem.M.norm(newx, newgrad);
    %    
    %    % Make sure we don't use too much memory for the store database
    %    storedb.purge();
    %    
    %    % Transfer iterate info
    %    x = newx;
    %    key = newkey;
    %    cost = newcost;
    %    grad = newgrad;
    %    gradnorm = newgradnorm;
    %    
    %    % iter is the number of iterations we have accomplished.
    %    iter = iter + 1;
    %    
    %    % Log statistics for freshly executed iteration
    %    stats = savestats();
    %    info(iter+1) = stats;
    %    
    %end
    %
    %
    %info = info(1:iter+1);

    %if options.verbosity >= 1
    %    fprintf('Total time is %f [s] (excludes statsfun)\n', ...
    %            info(end).time);
    %end

    %% ----------- GD iterations ---------------

    %% ------------nested function for managing/saving the stats
    % function stats = savestats()
    %     stats.iter = iter;
    %     stats.cost = cost;
    %     stats.gradnorm = gradnorm;
    %     if iter == 0
    %         stats.stepsize = NaN;
    %         stats.time = toc(timetic);
    %         stats.linesearch = [];
    %     else
    %         stats.stepsize = stepsize;
    %         stats.time = info(iter).time + toc(timetic);
    %         stats.linesearch = lsstats;
    %     end
    %     stats = self.statsfun(X, stats, itersdb);
    % end
    %% ------------nested function -----------------------------
end

