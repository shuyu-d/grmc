classdef GRMC < Problem
% Class of the graph-regularized matrix completion problem 
%
% Reference:
% 
% S. Dong, P.-A. Absil, and K. A. Gallivan. Riemannian gradient descent
% methods for graph-regularized matrix completion. Linear Algebra and its Applications, 2020. 
% DOI: https://doi.org/10.1016/j.laa.2020.06.010.
% 
%
% Contact: Shuyu Dong (shuyu.dong@uclouvain.be), ICTEAM, UCLouvain.
% 
% Latest version: September, 2021.
     
    properties        
        params_pb;
        mcinfo;
        stepsz_estimator;
        statsfun;
        stopfun;
        tid; 
    end
    properties (Constant)
        default_pb = struct('rank', 10, ...
                            'alpha_r', 0.05, 'alpha_c', 0.05,...
                            'Lreg_gammar', 0, 'Lreg_gammac', 0,...
                            'delta_rgrad', 1e-7);

        NAMES_MAN  = {'precon'};

        KEYS_TCscores = {'RMSE','MSE','NRMSE','ND','MAPE',...
            'RMSE_tr','MSE_tr','NRMSE_tr','ND_tr','MAPE_tr',...
            'RMSE_t','MSE_t','NRMSE_t','ND_t','MAPE_t',...
            'relErr', 'SNR'};
        DEFAULT_rperi_names = {'niter', 'time', 'RMSE_t', 'RMSE', 'RMSE_tr', 'relErr', 'gradnorm', 'cost'};
    end
    methods (Static)
        function rowid = get_rowid_TCscores(key)
            rowid =  find(strcmp(key, GRMC.KEYS_TCscores));
        end
        function rowid = get_rowid_resPerIter(key)
            rowid =  find(strcmp(key, GRMC.DEFAULT_rperi_names));
        end
        function p = build_parser(paramset, p)
            % Build an input argument parser for a given struct of parameters.
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
        function SNR = computeSCORES_SNR(Xsol, Xclean)
            sn_ratio = sum(Xclean(:).^2) / sum((Xsol(:)-Xclean(:)).^2);
            SNR = 10 * log10(sn_ratio);
        end
        function TCscores = computeSCORES_TC(Delta, Ttar, rmse)
            % From 1 to 5: RMSE, MSE, NRMSE, ND, MAPE
            %
            % input: Delta,  Ttar must be of size [_ x 1].
            if nargin < 3
                MSE = mean(Delta.^2);
                rmse = sqrt(MSE);
            else
                MSE = rmse^2;
            end
            % mtemp2 = mean(abs(Ttar));
            
            TCscores = [rmse;...
                MSE;...
                nan;...
                nan;...
                nan ];
        end
        function mcscores = compute_NRMSEscores(mcscores, Ttar, Ttar_tr, Ttar_t)
            %
            rowid = [3; 8; 13];
            rowid_rmse = [1; 6; 11];
            mtemp2 = [mean(abs(Ttar(:))); mean(abs(Ttar_tr)); ...
                mean(abs(Ttar_t))];
            mcscores(rowid) = mcscores(rowid_rmse)./mtemp2;
        end
        function relErr = computeSCORES_relErr(mse, mcinfo)
            % Compute the relative error: |X-Xtar|_F / |Xtar|_F.
            distF = sqrt( (numel(mcinfo.I)+numel(mcinfo.Itest))*mse ) ;
            relErr = distF/sqrt(sum([mcinfo.Xtr;mcinfo.Xte].^2));
        end
        function stop  = stopfun_rmse(optpb, X, info, last)
            stop = false;
            if last>1 && info(last).RMSE_tr < optpb.params_opt.tol_rmse %           
                  stop = true;
            end
        end
        function stop  = stopfun_manopt(optpb, X, info, last)
            stop = false;
            if last>1 && info(last).gradnorm > 1e7 %           
                  stop = true;
            end
        end
        function stop  = stopfun_ml1m(optpb, X, info, last)
            stop = false;
            if last>1 && info(last).gradnorm > 1e7 %           
                  stop = true;
            end
            % Relative change in RMSE-tr drops below a certain tolerance after long enough time.
            % This criterion will be effective if the time budget criterion is not imposed (or is too large).
            if isfield(info(last), 'RMSE_tr') && isfield(optpb.params_opt, 'tol_relchg_rmse')
                if info(last).time > 10 && last > 10 
                    rmses = [info(last-6:last).RMSE_tr];
                    relchg = max( abs(diff(rmses))./rmses(1:end-1) );

                    % relchg = abs(info(last).RMSE_tr - info(last-1).RMSE_tr)/info(last).RMSE_tr;
                    if relchg < optpb.params_opt.tol_relchg_rmse
                        fprintf('....Stopping criterion ML1M triggered: Tol relchg-rmse reached\n'); 
                        stop = true;
                    end
                end
            end 
        end
        function val = compute_rperi_relErr(rmse_all, mcinfo)
            val = rmse_all /...
                sqrt( mean([mcinfo.Xtr;mcinfo.Xte].^2) ) ;
        end
        function val = compute_rperi_RMSE(Delta)
            val = sqrt(mean(Delta.^2));
        end
        function [stats] = statsfun_cv(optpb, U, stats, store, mcinfo)
            stats.RMSE_t  = NaN ;
            stats.RMSE    = NaN ;
            stats.RMSE_tr = NaN ;
            stats.relErr  = NaN ; 
        end
        function [stats] = statsfun_unified_(optpb, X, stats, store, mcinfo)
            if isfield(X,'G')
                G = X.G; H = X.H;
            elseif isfield(X,'R')
                G = X.L; H = X.R;
            elseif isfield(X,'U')
                G = X.U * X.S;
                H = X.V; 
            end
            if ~isfield(store,'PX')
                store.PX = spmaskmult(G, H', mcinfo.I, mcinfo.J);
            end
            if ~isfield(store,'PX_t')
                store.PX_t = spmaskmult(G, H', mcinfo.Ite,...
                                            mcinfo.Jte);
            end
            if ~isfield(store,'err_Omega')
                store.err_Omega = store.PX - mcinfo.Xtr;
            end
            rmse_tr = sqrt(mean(store.err_Omega.^2));
            rmse_te = sqrt(mean((store.PX_t-mcinfo.Xte).^2));
            ntr     = mcinfo.sz_tr ;
            nte     = mcinfo.sz_te ;
            mse_all     = (rmse_tr^2 *ntr + rmse_te^2 *nte)/(ntr+nte) ;
            rmse_all    = sqrt( mse_all );
            relErr      =  rmse_all / sqrt( mean([mcinfo.Xtr;mcinfo.Xte].^2) ) ;             
 
            stats.RMSE_t  = rmse_te ; 
            stats.RMSE    = rmse_all ;
            stats.RMSE_tr = rmse_tr ; 
            stats.relErr  = relErr ;
        end

        
        function mcinfo = build_mcinfo_from_sparse_matrix(spMat, SR)
            % INPUT 
            % spMat:        a MATLAB (sparse) matrix 
            % SR:           sampling rate 

            if nargin < 2 
                SR = 0.8; 
            end

            [c1,c2, vals] = find(spMat);
            dims = size(spMat); 
            ntotal = prod(dims); 
            nz = numel(vals);
            
            N = ceil(SR*nz);
            s_tr = randsample(nz, N);
            % s_tr = sort(s_tr);
            ntr  = numel(s_tr); 
            
            s_comp = setdiff([1:nz], s_tr);
            ncomp  = numel(s_comp); 
            
            s_te_comp = s_comp(randsample(ncomp, min(ncomp,ntr)));
            s_te = s_comp(randsample(ncomp, max(20, ceil(min(ncomp,ntr/8))) ) );
            
            if size(s_te,2) > 1 
                s_te = s_te';
            end
            if size(s_te_comp,2) > 1 
                s_te_comp = s_te_comp';
            end
            
            I = c1(s_tr);
            J = c2(s_tr);
            
            Ite = c1(s_te);
            Jte = c2(s_te);
            
            Ite_comp = c1(s_te_comp);
            Jte_comp = c2(s_te_comp);
            
            Xtr = vals(s_tr); 
            Xte = vals(s_te); 
            Xte_comp = vals(s_te_comp); 

            mcinfo = struct('I', uint32(I), 'J', uint32(J), ...
                'Ite', uint32(Ite), 'Jte', uint32(Jte), ...    
                'Ite_comp', uint32(Ite_comp), 'Jte_comp', uint32(Jte_comp), ...    
                'Xtr', Xtr, 'Xte', Xte,...
                'Xte_comp', Xte_comp,...
                'SR',  SR, 'sz_tr', numel(I), 'sz_te',numel(Ite),...
                'sz_te_comp', numel(Ite_comp), ... 
                'spMat', spMat, ... 
                'size_M', dims); 
        end

        function [f, store] = cost_twofac_grmf(X, store, mcinfo, params_pb, L)
        % Compute the cost function 
            if ~isfield(store, 'val')
                if ~isfield(store,'PX')
                    store.PX = spmaskmult(X.L, X.R', mcinfo.I, mcinfo.J);
                end
                if ~isfield(store,'err_Omega')
                    store.err_Omega = store.PX - mcinfo.Xtr;
                end
                if ~isfield(store,'LG')
                    store.LG = params_pb.alpha_r*(X.L + params_pb.Lreg_gammar*L.Lr * X.L);
                end
                if ~isfield(store,'LH')
                    store.LH = params_pb.alpha_c*(X.R + params_pb.Lreg_gammac*L.Lc * X.R); 
                end
                store.val = (.5*sum(store.err_Omega.^2) ...
                                + .5*trace(X.L'*store.LG) ... 
                                + .5*trace(X.R'*store.LH));
            end
            f = store.val;
        end
        function [grad, store] = rgrad_precon_mleqk(X, store, mcinfo, params_pb, L)
        % Compute the gradient 
            if ~isfield(store, 'grad')
                m = size(X.L,1); n = size(X.R,1);
                if ~isfield(store,'PX')
                    store.PX = spmaskmult(X.L, X.R', mcinfo.I, mcinfo.J);
                end
                if ~isfield(store,'err_Omega')
                    store.err_Omega = store.PX - mcinfo.Xtr;
                end
                if ~isfield(store,'LG')
                    store.LG = params_pb.alpha_r*(X.L + params_pb.Lreg_gammar*L.Lr * X.L);
                end
                if ~isfield(store,'LH')
                    store.LH = params_pb.alpha_c*(X.R + params_pb.Lreg_gammac*L.Lc * X.R); 
                end
                if ~isfield(store, 'GtG')
                    store.GtG = (X.L') * X.L;
                end
                if ~isfield(store, 'HtH')
                    store.HtH = (X.R') * X.R;
                end

                S = sparse(double(mcinfo.I), double(mcinfo.J), store.err_Omega, m, n );
                store.grad = struct('L', (S*X.R + store.LG)/store.HtH,...
                                    'R', (S'*X.L+ store.LH)/store.GtG);
            end
            grad = store.grad ;   
        end
         
        function [tmin, store] = linesearch_initialpt(X, dir, store, mcinfo, params_pb, L)
        % Compute the initial step-size by exact line search in the tangent
        % space (and NOT on the retracted curve). This is only used as an
        % initial guess. 
            
            if ~isfield(X,'G')
                G = X.L; H = X.R;
                dirG = dir.L; dirH = dir.R;
            else
                G = X.G; H = X.H;
                dirG = dir.G; dirH = dir.H;
            end
            if ~isfield(store,'PX')
                store.PX = spmaskmult(G, H', mcinfo.I, mcinfo.J);
            end
            if ~isfield(store,'err_Omega')
                store.err_Omega = store.PX - mcinfo.Xtr;
            end
            if ~isfield(store, 'GtG')
                store.GtG = (G') * G;
            end
            if ~isfield(store, 'HtH')
                store.HtH = (H') * H;
            end
            if ~isfield(store,'LG')
                store.LG = params_pb.alpha_r*(G + params_pb.Lreg_gammar*L.Lr * G);
            end
            if ~isfield(store,'LetaG')
                store.LetaG = params_pb.alpha_r*(dirG + params_pb.Lreg_gammar*L.Lr * dirG);
            end
            if ~isfield(store,'LH')
                store.LH = params_pb.alpha_c*(H + params_pb.Lreg_gammac*L.Lc * H); 
            end
            if ~isfield(store,'LetaH')
                store.LetaH = params_pb.alpha_c*(dirH + params_pb.Lreg_gammac*L.Lc * dirH);
            end
            
            if ~isfield(store,'pomega_xeta')
                % store.pomega_xeta = spmask( G*(dirH') + dirG *(H'), mcinfo.I, mcinfo.J);
                store.pomega_xeta = spmaskmult(G, dirH', mcinfo.I,mcinfo.J) + ...
                                    spmaskmult(dirG, H', mcinfo.I, mcinfo.J);
            end
            if ~isfield(store,'pomega_eta')
                % store.pomega_eta = spmask( dirG*(dirH') , mcinfo.I, mcinfo.J);
                store.pomega_eta = spmaskmult(dirG, dirH', mcinfo.I,mcinfo.J);
            end

            if ~isfield(store,'etaGtLG')
                store.etaGtLG = (dirG') * store.LG;
            end
            if ~isfield(store,'etaGtLetaG')
                store.etaGtLetaG = (dirG') *store.LetaG;
            end

            if ~isfield(store,'etaHtLH')
                store.etaHtLH = (dirH') * store.LH;
            end
            if ~isfield(store,'etaHtLetaH')
                store.etaHtLetaH = (dirH') *store.LetaH;
            end

            c1 = (store.err_Omega') * store.pomega_xeta + ...
            trace(store.etaGtLG + store.etaHtLH);
            
            c2 = 0.5*sum(store.pomega_xeta.^2) + (store.err_Omega')*store.pomega_eta +...
                0.5*trace(store.etaGtLetaG + store.etaHtLetaH);
            
            c3 = (store.pomega_xeta')* store.pomega_eta ;

            c4 = 0.5*sum(store.pomega_eta.^2) ;
            
            c = [c1,c2,c3,c4];
            if ~any(isnan(c))
                ts = roots(fliplr([1:4].*c));
                % ts = ts(isreal(ts));
                ts = ts(imag(ts)==0);
                ts = ts(ts>0);
                
                if isempty(ts)
                    ts = 1;
                end
                dfts = polyval([c4 c3 c2 c1 0], ts);
                [~, iarg] = min(dfts);
                tmin = ts(iarg);
            else
                tmin = 10;
            end
        end
 

        function [man, costfun, gradfun, hessfun, precon, lsfun] = construct_pb(manifoldname, mcinfo, params_pb, L)
            switch manifoldname
                case 'precon'
                    man     =  productmanifold( struct('L', euclideanfactory(mcinfo.size_M(1), params_pb.rank), 'R', euclideanfactory(mcinfo.size_M(2), params_pb.rank)) );                    
                    costfun =  @(X, store) GRMC.cost_twofac_grmf(X, store, mcinfo, params_pb, L); 
                    gradfun =  @(X, store) GRMC.rgrad_precon_mleqk(X, store, mcinfo, params_pb, L); 
                    lsfun   =  @(X, dir, store) GRMC.linesearch_initialpt(X, dir, store, mcinfo, params_pb, L); 
                    stopfun =  @(optpb, X, info, last) GRMC.stopfun_ml1m(optpb, X, info, last) ;    
                    
                otherwise
                    error(sprintf('%s is not implemented yet in Manopt..\n', manifoldname) );
            end
            if ~exist('hessfun', 'var')
                hessfun = [];
            end
            if ~exist('precon', 'var')
                precon = [];
            end
            man.name    = @() manifoldname;
        end
    end % (methods (Static))
    
    methods
        function self = GRMC(mcinfo, rank, params, Lr, Lc, manifoldname)
            % INPUT 
            %   mcinfo:         Unified matrix completion input data in COO format
            %   rank:           The rank parameter of the matrix factorization model 
            %   params:         Regularization parameters and the metric parameter delta
            %                       (1-2): alpha_r, alpha_c
            %                       (3-4): gamma_r, gamma_c
            %                       (5)  : delta (in the metric) 
            % 
            %   Lr, Lc:         The graph Laplacian matrices 
            %   manifoldname:   Name of the search space endowed with a certain metric  

            dims = mcinfo.size_M;  

            if nargin < 3, params_pb = GRMC.default_pb; end
            if nargin < 4, Lr = sparse(dims(1)); end
            if nargin < 5, Lc = sparse(dims(2)); end
            if nargin < 6, manifoldname = GRMC.NAMES_MAN{1}; end
            
            %% Construct struct of the problem parameters
            params_pb = struct('rank', rank, ...
                            'alpha_r', params(1), ...  
                            'alpha_c', params(2),...
                            'Lreg_gammar', params(3), ... 
                            'Lreg_gammac', params(4),...
                            'delta_rgrad', params(5));
            L = struct('Lr', Lr, 'Lc', Lc); 
            %% Construct function handles.
            [man, costfun, gradfun, hessfun, precon, lsfun] = GRMC.construct_pb(manifoldname, mcinfo, params_pb,L); 

            %% Initialize class properties.
            self = self@Problem(man, costfun, gradfun,[],hessfun, []);
            
            self.params_pb  = params_pb;

            self.mcinfo     = mcinfo;
            self.stepsz_estimator = lsfun;
            self.stopfun = @(optpb, X, info, last)GRMC.stopfun_manopt(optpb, X, info, last);
            self.statsfun = @(optpb, X, stats, store)GRMC.statsfun_unified_(optpb, X, stats, store, mcinfo);
        end

        % Generate initial point
        function X0 = initialization(self, method_init, rank_feat)
        % INPUT
        % method_init: A string indicating the initialization method; or a given point X.
            if nargin < 3
                rank_feat = self.params_pb.rank;
            end
            if ischar(method_init)
                switch method_init
                    case 'rand1'
                        D0 = GRLRMC.default_pb.C0*self.pbinfo.mstar_frob2; 
                        P = randn(self.data.dims(1),self.params_pb.rank);
                        Q = randn(self.data.dims(2),self.params_pb.rank);
                        hatM = P*(Q');
                        fa = sum(hatM(:).^2)/self.pbinfo.mstar_frob2; 
                        hatM = hatM / sqrt(fa);  
                        [U, S, V] = svds(hatM, rank_feat); 
                    case 'Mmean'
                        val = GRLRMC.default_pb.C0*sqrt(mean(self.mcinfo.Xtr.^2)); 
                        hatM = sparse(double(self.mcinfo.I), double(self.mcinfo.J), ...
                                      self.mcinfo.Xtr-val,...
	                                  self.mcinfo.size_M(1), self.mcinfo.size_M(2) ) +...
                               val*randn(self.mcinfo.size_M);
                        fa = sum(hatM(:).^2)/self.pbinfo.mstar_frob2; 
                        hatM = hatM / sqrt(fa);  
                        [U, S, V] = svds(hatM, rank_feat);
                    case 'random'
                        P = randn(self.data.dims(1),self.params_pb.rank);
                        Q = randn(self.data.dims(2),self.params_pb.rank);
                        [U,S,V] = svds(P*(Q'),self.params_pb.rank); 
                        C = mean(self.mcinfo.Xtr.^2);
                        S = S* sqrt(C/rank_feat);
                    case 'M0_unbalanced'
                        fa = 5; 
                        [U, S, V] = svds(sparse(double(self.mcinfo.I),...
                                                double(self.mcinfo.J),...
                                                self.mcinfo.Xtr,...
                                                self.mcinfo.size_M(1),...
                                                self.mcinfo.size_M(2)), ...
                                        rank_feat);
                        U = U * fa; V = V / fa; 
                    case 'M0'
                    % default init method: 
                    [U, S, V] = svds(sparse(double(self.mcinfo.I),...
                                            double(self.mcinfo.J),...
                                            self.mcinfo.Xtr,...
                                            self.mcinfo.size_M(1),...
                                            self.mcinfo.size_M(2)), ...
                                     rank_feat);
                    % otherwise
                end
                name_manifold = self.manifold.name();
                switch name_manifold
                  case {'precon'}  
                    sqrtS = diag(sqrt(diag(S)));
                    X0    = struct('L', U*sqrtS, 'R', V*sqrtS);
                  otherwise
                    error('initialization in other manifold structures not implemented yet \n');
                end
            elseif isstruct(method_init)
                % This is when method_init is a given point X in the search space. 
                X0 = method_init;
            else 
                error('The input argument method_init is not compatible...\n');
            end
        end


        function set_samplrate(self, SR)
        % This function refreshes the problem class by generating a new
        % set of revealed entries at the given sampling rate SR.
            if nargin < 1
                SR = 0.2;
            end
            mcinfo = GRMC.build_mcinfo_from_sparse_matrix(self.mcinfo.spMat, SR);
            self.refresh_obj([], mcinfo);
        end
        function loadnew_paramspb(self, params_pb)
        % Set up parameter values by a given struct of parameters.
            self.refresh_obj([], [], params_pb);
        end
        function set_params_pb(self, varargin)
        % Set up parameter values by the pair of keyword and value. The parameters are
        % 1. paramd.
        % 2. rank. Note that the parameter sampl_rate is related to the given data.
        % Warning: the parser only admits keywords prescribed in GRMC.default_pb.
            params_pb = self.params_pb;
            parser = GRMC.build_parser(params_pb);
            parse(parser, varargin{:});
            fds = fieldnames(parser.Results);
            for i = 1 : length(fds)
                params_pb.(fds{i}) = parser.Results.(fds{i});
            end
            self.refresh_obj([], [], params_pb);
        end
        function refresh_obj(self, manifold_name, mcinfo, params_pb)
            if nargin < 2
                manifold_name = [];
            end
            if nargin < 3
                mcinfo = [];
            end
            if nargin < 4
                params_pb = [];
            end
            if isempty(manifold_name)
                manifold_name = self.manifold.name();
            end
            if isempty(mcinfo)
                mcinfo = self.mcinfo;
            end
            if isempty(params_pb)
                params_pb = self.params_pb;
            end
            %% Reconstruct function handles.
            [man, costfun, gradfun, hessfun, precon, lsfun] = GRMC.construct_pb(manifold_name, mcinfo, params_pb);
            
            %% Reload class properties.
            self.refresh_obj@Problem(man, costfun, gradfun,[],hessfun, []);
            self.stepsz_estimator = lsfun;
            self.params_pb = params_pb;
            self.mcinfo = mcinfo;
        end
    end
end

