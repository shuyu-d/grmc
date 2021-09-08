classdef Solver < handle
% One method is identified with one instance of the Solver object. 
% - problem class (problem-related parameters, manifold structure, metric, cost, gradient etc), 
% - optimization-related parameters,
% - method name. 
    properties
        problem; 
        params_opt;
        % A container of limited length. A struct containing (1) iterid; (2) arrays
        % of structs such as x, rinfo, lsinfo, etc.  
        dbiters;
        % stats; 
    end
    properties (Constant)
        NAMES_ALGO = {'GRALS';...
                      'GRALS1';...
                      'GRALS2';...
                      'RSD_MANOPT';...
                      'RCG_MANOPT';...
                      'RCGprecon_MANOPT'; ... % 'RCGCruderls_MANOPT';...
                      'RSD'; ...
                      'RCG'} ; 
        NAMES_LS = {'lsArmijo'; 'linemin'; 'lsBB'}; 
        NAMES_SS0 = {'exact'; 'one'; 'lipschitz'}; 
        DEFAULT_OPT =   struct('method_init', 'M0', ...
                            'maxtime', 50, ...
                            'maxiter', 500, ...
                            'rcg_usePrecon', false,...
                            'grmf_maxiter', 100,...
                            'grmf_maxiter_cg', 30,...
                            'grmf_eps', 1e-7,...
                            'tolgradnorm', 1e-10, ...
                            'tol_relchg_rmse', 1e-10, ...
                            'tol_rmse', 1e-10, ...
                            'stepsize0_type', 'exact', ...  % {one, exact, lipschitz, BB} 
                            'ls_backtrack', true,...
                            'ls_force_decrease', true,...
                            'minstepsize', 1e-20,...
                            'useRand', false, ...
                            'storedepth', 5, ...
                            'compute_statsPerIter', true,...
                            'verbosity', 2);

    end
    methods (Static)
        [alpha0, itersdb] = stepsize_init_lipschitz(manifold, costfun, X, d, itersdb)
        function p = build_parser(paramset, p)
        % Build a parser for a structure paramset.
            if nargin < 2
                p = inputParser;
            end
            fds = fieldnames(paramset);
            for i = 1 : length(fds)
                if ~strcmp(fds{i},'Properties')
                    addParameter(p, fds{i}, paramset.(fds{i}));
                end
            end
            p.KeepUnmatched = true;
        end
        function params = parseArgs(varargin)
		    % Input argument parser. 
            p = Solver.build_parser(Solver.DEFAULT_OPT);
            parse(p, varargin{:});
            fds = fieldnames(Solver.DEFAULT_OPT);
		    for i = 1 : length(fds)
		    	params.(fds{i}) = p.Results.(fds{i});
		    end
		end 
    end
    methods
        function self = Solver(problem, varargin)
        % % A container of limited length. A struct containing (1) iterid; (2) arrays
        % % of structs such as x, rinfo, lsinfo, etc.  
        % db_iters;
            if nargin < 2
                params_opt = Solver.DEFAULT_OPT; 
            end
            params_opt = Solver.parseArgs(varargin{:}); 
            self.problem = problem; 
            self.params_opt = params_opt;
            self.dbiters = struct('clen', 20, 'iterid', 0, 'x', [], 'rinfo',...
                                  [], 'lsinfo', []);
        end
        function refresh_obj(self, pb, params)
            if nargin < 2
                pb = [];
            end
            if nargin < 3
                params = [];
            end
            if isempty(pb)
                pb = self.problem;
            end
            if isempty(params)
                params = self.params_opt;
            end
            %% Reload properties 
            self.problem = pb; 
            self.params_opt = params;
            fprintf('Solver properties reloaded...\n');
        end
        function self = loadnew_paramsopt(self, params_opt)
            self.params_opt = params_opt;
        end
        function set_params(self, varargin)
        % Set up parameter values by the pair of keyword and value. The parameters are
        % 1. paramd. 
        % 2. rank. Note that the parameter sampl_rate is related to the given data. 
        % Warning: the parser only admits keywords prescribed in LRTC.default_pb. 
            opts   = self.params_opt; 
            parser = Solver.build_parser(opts);
            parse(parser, varargin{:});
            fds = fieldnames(parser.Results);
            for i = 1 : length(fds)
                opts.(fds{i}) = parser.Results.(fds{i});
            end
            self.refresh_obj([], opts);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function X0 = initialization(self, method_init)
            if isstruct(method_init) 
                % This is when method_init = X (a matrix product variable)
                X0 = method_init;
            elseif iscell(method_init) && isprop(self.problem, 'tcinfo') 
                % Convert CPD formats with the k factor matrices in cells into the unified format in the CPD 
                % product space: x = (x.u1, x.u2, x.u3).  
                for i = 1 : numel(self.problem.tcinfo.unames)
                    X0.(self.problem.tcinfo.unames{i}) = method_init{i};
                end
            else 
                error('The initial point format is not supported ...\n'); 
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                       Solvers in separate files                            % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [newx, newf, lsstats] = linesearch_none(self, X, d, itersdb)
        [newx, newf, lsstats] = linesearch_Armijo(self, x, d, f0, ...
                                                  df0, options, store, stepsize0)
        [X, stats, self] = solve_RGD_lsArmijo(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsFree(self, method_init, varargin)
        [X, stats, self] = solve_RCG_lsFree(self, method_init, varargin)
        [X, stats, self] = solve_accRGD_(self, method_init, varargin)
        [X, stats, self] = solve_accRGD_rs1(self, method_init, varargin)
        
        [X, stats, self] = solve_RGD_lsBB(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsBB1(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsBB2(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsRBB(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsRBB1(self, method_init, varargin)
        [X, stats, self] = solve_RGD_lsRBB2(self, method_init, varargin)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %           Methods for producing curves, figures and tables.                % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        curve = extract_rperi(self, stats, y, x, varargin)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                           Solvers from MANOPT                              % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [X, stats, self] = solve_rsd_manopt(self, method_init, varargin)
            if ischar(method_init)
                % override by the option in params_opt (this is set at the
                % instantiation of self, via the key string `methods_init`). 
                method_init = self.params_opt.method_init;
            end
            X0 = self.initialization(method_init); 
            methodname = Solver.NAMES_ALGO{3}; %RSD_MANOPT

            optproblem = struct('Xtar', self.problem.Tstar, 'Xtar_r', [],...
                              'M', self.problem.manifold, 'cost',...
                              self.problem.cost, 'grad', self.problem.grad);
            % if ~isempty(self.hess)
            %     optproblem.hess = self.hess;
            % end
            % if ~isempty(self.lsstepsize_initialguess)
            %     optproblem.linesearch = self.lsstepsize_initialguess;
            % end
            % if ~isempty(self.preconditioner) && self.params_opt.rcg_usePrecon 
            %     optproblem.precon = self.preconditioner;
            %     methodname = 'RSDprecon_MANOPT';
            % end

            options_manopt          = self.params_opt;
            options_manopt.statsfun = self.problem.statsfun;
            if ~isempty(self.problem.stopfun)
                options_manopt.stopfun = self.problem.stopfun;
            end
            [X, ~, stats] = steepestdescent(optproblem, X0, options_manopt) ;          
        end 
        function [X, stats, self] = solve_rcg_manopt(self, method_init, varargin)
            if ischar(method_init)
                % override by the option in params_opt (this is set at the
                % instantiation of self, via the key string `methods_init`). 
                method_init = self.params_opt.method_init;
            end
            X0 = self.initialization(method_init); 
            % note (#702): add to `self.params_opt` a function handle `linesearch`, then the linesearch 
            methodname = Solver.NAMES_ALGO{2};
            % funtion will be `linsearch_hint.m`.
            optproblem = struct('Xtar', self.problem.Tstar, 'Xtar_r', [],...
                              'M', self.problem.manifold, 'cost', ...
                              self.problem.cost, 'grad', self.problem.grad);
            if ~isempty(self.problem.hess)
                optproblem.hess = self.hess;
            end
            optproblem.params_opt = self.params_opt; % in case the user-defined stopping criterions are used, see "manopt/stoppingcriterion.m".
            % if ~isempty(self.problem.stepsz_estimator)
            %     optproblem.linesearch = self.problem.stepsz_estimator; %self.lsstepsize_initialguess;
            % end
            % if ~isempty(self.preconditioner) && self.params_opt.rcg_usePrecon 
            %     optproblem.precon = self.preconditioner;
            %     methodname = 'RCGprecon_MANOPT';
            % end

            options_manopt          = self.params_opt;
            options_manopt.statsfun = self.problem.statsfun;
            if ~isempty(self.problem.stopfun)
                options_manopt.stopfun = self.problem.stopfun;
            end
            [X, ~, stats] = conjugategradient(optproblem, X0, options_manopt) ;          
        end 
        function [X, stats, self] = solve_rtr_manopt(self, method_init, varargin)

            if nargin < 2
                method_init = 'M0';
            end
            methodname = 'RTR_MANOPT'; %Solver.NAMES_ALGO{4};
            X0 = self.initialization(method_init);
            self.optlog.X0 = X0;
            % note (#702): add to `self.params_opt` a function handle `linesearch`, then the linesearch 
            % funtion will be `linsearch_hint.m`.
            optproblem = struct('Xtar', self.Xstar, 'Xtar_r', self.Xstar_r,...
                                'M', self.problem.manifold, 'cost', self.cost, 'grad', self.grad);
            if ~isempty(self.hess)
                optproblem.hess = self.hess;
            end
            if ~isempty(self.lsstepsize_initialguess)
                optproblem.linesearch = self.lsstepsize_initialguess;
            end
            % if ~isempty(self.preconditioner)  
            %     optproblem.precon = self.preconditioner;
            % end
             options_manopt          = self.params_opt;
            options_manopt.statsfun = self.statsfun;
            [X, ~, stats] = trustregions(optproblem, X0, options_manopt) ;          
            self.optlog.(methodname) = stats;
            self.collect_results(X, stats, 'methodname', methodname, varargin{:});
        end 
        %=========================================================================
        % note (#226): steepest descent using BB-stepsize
        %-------------------------------------------------------------------------
        % This part is NOT urgent. However, in [Liu et al. 2016: Trace-penalty
        % min.], Sec.3.2, it is mentioned that "restarting strategy" can
        % accelerate convergence: 
        % "However, a typical behavior of
        % gradient methods is that they can reduce the objective function rather rapidly at 
        % an initial
        % stage, but the amount of reduction can become extremely small as iterates get closer to
        % a solution. In trace-penalty minimization, it has been observed that restarting 
        % the gradient
        % method with a modified X can usually help accelerate convergence and achieve a higher
        % accuracy more quickly. " 
        function [newx, newf, itersdb] = linesearch_BBstep(self, X, d, itersdb)
        % INPUT
        %       itersdb: cost, grad, gradnorm, X_old, grad_old,
        %       store(for-computing-cost)
        % OUTPUT
        %       alpha, newx, itersdb
        % This function is called after (grad^t) is computed in the main
        % algorithm. 

            % (0) local parameters
            %     Backtracking default parameters. These can be overwritten in the
            %     options structure which is passed to the solver.
            default_options.ls_contraction_factor = .5;
            default_options.ls_suff_decr = 1e-4;
            default_options.ls_max_steps = 5;
            default_options.ls_backtrack = true;
            default_options.ls_force_decrease = false;
            options = mergeOptions(default_options, self.params_opt);
            
            contraction_factor = options.ls_contraction_factor;
            suff_decr = options.ls_suff_decr;
            max_ls_steps = options.ls_max_steps;
  
            % (1) Compute the BB stepsize
            if ~isfield(itersdb, 'X_old')
                % This is the case iter=1:
                alpha = self.lsstepsize_initialguess(X, dir, struct());
            else
                % This is the case iter > 1, store has `{X_old, grad_old}` 
                S = self.problem.manifold.lincomb(X,1,X,-1,itersdb.X_old);
                Y = self.problem.manifold.lincomb(X,1,itersdb.grad, -1, itersdb.grad_old);
                if rand < 0.5
                    alpha = man.inner(X,S,Y)/man.norm(X,Y)^2; 
                else
                    alpha = man.norm(X, S)^2/man.inner(X,S,Y); 
                end
            end
            
            % (1) Backtracking 
            %------ Armijo backtracking -----------
            % a. unload {f0=f(X), df0= manifold.inner(x, grad, desc_dir) }
            f0 = itersdb.f0; df0 = itersdb.df0; 
            % Make the chosen step and compute the cost there.
            newx = problem.M.retr(X, d, alpha);
            % note (#227): intermediate info. (GH,err_Omega..) for computing
            %              newf inside line search is NOT interesting for the 
            %              outer-loops. No need to be cached. 
            % newkey = storedb.getNewKey();
            % newf = getCost(problem, newx, storedb, newkey);
            [newf, store] = self.cost(newx, struct());
            cost_evaluations = 1;
            
            % Backtrack while the Armijo criterion is not satisfied
            while options.ls_backtrack && newf > f0 + suff_decr*alpha*df0
                
                % Reduce the step size,
                alpha = contraction_factor * alpha;
                
                % and look closer down the line
                newx = problem.M.retr(x, d, alpha);
                % newf = getCost(problem, newx, storedb, newkey);
                [newf, store] = self.cost(newx, struct());
                cost_evaluations = cost_evaluations + 1;
                
                % Make sure we don't run out of budget
                if cost_evaluations >= max_ls_steps
                    break;
                end
            end
            % todo: #p2 (#227) pair with nonmonotone line search 
            % The following block is not reached, ls_force_decrese==false here.
            if options.ls_force_decrease && newf > f0
                alpha = 0;
                newx = x;
                newkey = key;
                newf = f0; %#ok<NASGU>
            end
            %------ Armijo backtracking -----------
            % update itersdb using the information for computing {newf}
            itersdb.store = store;
            itersdb.grad_old = itersdb.grad;
            itersdb.X_old = X;
            % Return some statistics also, for possible analysis.
            lsstats.costevals = cost_evaluations;
            lsstats.alpha = alpha;
        end
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
        %===================== GD-BBstepsize ====================================================
        function [h, info] = check_initialstepsize_sd(self, x)
        % function [curve, info] = check_initialstepsize(problem, x, dir, alpha0, roots, froots)
           % - input: x, dir, alpha_init, manifold.
           % - output: a 1-D curve "f(x(t)vs t)", where x(t) = R_x(t.dir), for t=[0, 2alpha_init]. Also show markers on the x-axis for the 3 roots found by `pbgrlrmc/func:linesearch_initialpt()`.
           % if nargin < 5
           %    % froots always come together with `roots`.
           %    roots = NaN;
           %    froots = NaN;
           % end
            problem=struct('M', self.problem.manifold, 'cost', self.problem.cost, 'grad', self.problem.grad);
            if ~isempty(self.problem.hess)
                problem.hess = self.problem.hess;
            end
            if ~isempty(self.problem.stepsz_estimator)
                problem.linesearch = self.problem.stepsz_estimator;
            end
            % problem = self.optproblem; 
            if nargin < 2
                x = problem.M.rand();
            end
           
            % Compute the value f0 at f and directional derivative at x along d.
            storedb = StoreDB();
            xkey = storedb.getNewKey();
            f0 = getCost(problem, x, storedb, xkey);
            temp = getGradient(problem, x, storedb, xkey); 
            dir = problem.M.lincomb(x, -1, temp); 
            h(1) = figure();
            checkgradient(problem, x, dir); 
           if isfield(problem, 'hess')
               h(3) = figure();
               % checkhessian(problem, x, dir);
               checkhessian(problem);
           end
           store = storedb.getWithShared(xkey);
           [alpha0, store] = problem.linesearch(x, dir, store); 
           % Compute the value of f at points on the geodesic (or approximation
           % of it) originating from x, along direction d, for stepsizes in a
           % range = [0, 2*alpha0]|norm_d|.
           ts = linspace(0, 2, 51)*alpha0;
           value = zeros(size(ts));
           for i = 1 : length(ts)
               y = problem.M.exp(x, dir, ts(i));
               ykey = storedb.getNewKey();
               value(i) = getCost(problem, y, storedb, ykey);
           end           
           % And plot it.
           h(2) = figure();
           plot(ts, value); hold on;
           % plot([0,roots], [f0,f0+dflin_roots], 'k+');
           plot(ts(26), value(26), 'r^');
            info = struct('ts', ts, 'fts', value);
        end
    end

end
