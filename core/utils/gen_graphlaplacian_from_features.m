function [output, t] = gen_graphlaplacian_from_features(feat, opts) 
   
% opts:     options containing fileds like 'methodname', 'epsNN_GaussianKernel_sigma', 'sparsity' 
   
    if nargin < 2
        opts = struct('methodname', 'epsNN-GaussianKernel',...
                      'sparsity', 0.1); 
    end

    fprintf('\n.....Start building Lmat from data with method="%s"... ', opts.methodname); 
    t0=tic;
    switch opts.methodname 
        case 'epsNN-GaussianKernel'
            [W , eps, sigma]= buildW_epsNN_GaussianKer(feat, opts);
            % Wc = sparse(size(feat.feat_c,1), size(feat.feat_c,1));
            opts.eps = eps;
            opts.sigma = sigma;
        otherwise
            error('The method %s for building graph Laplacian matrices is not available.\n', opts.methodname);
    end
    t = toc(t0);
    fprintf('Done: '); toc(t0);
    L =  computeLmat_fromW(W, opts) ; 

    output.L = L ;
    output.W = W ;
    output.opts_gen = opts;  

    % L = struct('Lr', computeLmat_fromW(Wr, opts), ...
    %            'Lc', computeLmat_fromW(Wc, opts),...
    %            'Wr', Wr, ...
    %            'feat_samplrate', feat.feat_samplrate,...
    %            'opts_buildLmat', opts); 


    function [Wsp, eps] = buildW_sparsify(W, sparsity)
    % This function compute eps such that the empirical cumulative distribution
    % of X={W[ij], (i,j)\in [m]x[m]} at eps equals (1-sparsity):
    %      p((i,j)\in [m]x[m]: W[i,j] < eps) = 1-sparsity. 
    % Then eps serves as the threshold value for keeping the largest adjacency
    % coefficents of W: W_sparse = W*(1-indicator_{(i,j): W[ij] < eps})
        if nargin < 2 
            sparsity = 0.1; 
        end

        % Exclude all self-loops in the graph adjacencies 
        W = W - diag(diag(W));

        % Compute the empirical CDF function 
        [cdf, w] = ecdf(W(:));

        % Find eps such that ecdf(eps) = 1-sparsity, we target cdf=1-sparsity with a
        % tolerance radius of 1e-6 (if there are several candidates in the bin
        % of [cdf +- 1e-6], we take the one in the middle. 

        % Find the target bin of radius 2*tol (tol=1e-6) containing the
        % wanted eps: 
        tol = 1e-7; ids = [];
        while isempty(ids) 
            tol = 10*tol;
            ids = find( abs(cdf - (1-sparsity))<tol );
        end
        % Take the value in the middle of the targeted bin:
        len = numel(ids);
        eps = w(ids(ceil(len/2)));

        % Return the sparse graph adjacency matrix
        Wsp = sparse(W.*(W>=eps));
    end

    function [W_sp, eps, sigma] = buildW_epsNN_GaussianKer(featMat, opts)
        % `sigma` and `sparisty` are 2 independent parameters that control the properties of 
        % the wanted sparse adjacency matrix: sigma controls the variance of 
        % {W[ij], (i,j)\in [m]x[m]} and sparsity controls the sparse level of the sparsified 
        % W (c.f. `func: buildW_sparsify()` for its detailed usage).
        
        % sigma = opts.epsNN_sigma;

        % NOTE: use a value of sigma adaptive to the variance of the
        % entries in the distance matrix

        sparsity = opts.sparsity;
        Z = pdist2(featMat, featMat);
        sigma = var(Z(:))/5;

        [W_sp, eps] = buildW_sparsify(exp(-Z.^2/sigma), sparsity);
    end

    function Lmat = computeLmat_fromW(W, opts)
    % The computation depends on whether we choose to use the normalized graph Laplacian or just the combinatorial graph Laplacian.
        d = sum(W,1);
        if isfield(opts, 'type_graphLaplacian') && strcmp(opts.type_graphLaplacian,'normalized')
            Lmat = sparse(eye(size(W,1)) - diag(sqrt(d))*W*diag(sqrt(d)));
        else
            Lmat = sparse(diag(d)) - W;
        end
    end



end


