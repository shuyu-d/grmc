classdef Problem < handle 
% A generic problem class. 
% 
% References: [..]. 
% 
% Contact: Shuyu Dong (shuyu.dong@uclouvain.be), ICTEAM, UCLouvain.  
% Latest version: February 2020. 
% 
    properties
        manifold; 
        cost;
        grad;
        egrad; 
        hess;
        ehess; 
    end

    methods
        function self = Problem(manifold, cost, grad, egrad, hess, ehess)
            if nargin < 1, manifold = []; end
            if nargin < 2, cost = []; end
            if nargin < 3, grad = []; end
            if nargin < 4, egrad = []; end
            if nargin < 5, hess = []; end
            if nargin < 6, ehess = []; end

            %% Initialize class properties. 
            self.manifold = manifold; 
            self.cost     = cost;
            self.grad     = grad;
            self.egrad    = egrad;
            self.hess     = hess;
            self.ehess    = ehess;
            fprintf('All elements constructed...\n');
        end

        function refresh_obj(self, manifold, cost, grad, egrad, hess, ehess)
            %% Reload properties 
            self.manifold = manifold; 
            self.cost     = cost;
            self.grad     = grad;
            self.egrad    = egrad;
            self.hess     = hess;
            self.ehess    = ehess;
            fprintf('All problem properties reloaded...\n');
        end
        function res = checkgradient()
        end

    end


end

