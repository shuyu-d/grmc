function feat = gen_features_for_graphinfo(mcinfo, row_or_col, opts)
% 
% Construct feature matrices from rank-r SVD of M0: 
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
 

    if nargin < 3 
        opts = struct('hatM0_rank', 20); 
    end
    if strcmp(row_or_col , 'rows') 
        hatM0 = sparse(double(mcinfo.I),...
                   double(mcinfo.J),...
                   mcinfo.Xtr,...
                   mcinfo.size_M(1),...
                   mcinfo.size_M(2)); 
    else
        hatM0 = sparse(double(mcinfo.J),...
                   double(mcinfo.I),...
                   mcinfo.Xtr,...
                   mcinfo.size_M(2),...
                   mcinfo.size_M(1)); 
    end

    [U, S, V] = svds(hatM0, opts.hatM0_rank); 
    feat = U*S;
end

