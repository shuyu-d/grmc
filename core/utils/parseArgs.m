function params = parseArgs(param_name, varargin)
% Parse input arguments based on default parameter pairs. 
% 
% Contact: Shuyu Dong (shuyu.dong@uclouvain.be), ICTEAM, UCLouvain.
% 
% Latest version: September, 2021.
 

% Input argument parser 
% 
% param_name:   String used include 'runTester', 'gdat', 'rdat', 'opt', 'infoMethods2',
%                   'infoMethods'

    DEFAULT.realdata = struct('use_submatrix', false,...
                              'dims_submatrix', [300,600]);

    DEFAULT.buildLmat = struct('methodname', ...
                                {{ 'epsNN-GaussianKernel' }}, ...
                               'fromSideinfo', 0, ...
                               'hatM0_rank', 20, ...
                               'sparsity',  0.1);

    p = GRMC.build_parser(DEFAULT.realdata);
    p = GRMC.build_parser(DEFAULT.buildLmat, p);
    p = GRMC.build_parser(Solver.DEFAULT_OPT, p);

    parse(p, varargin{:});
    switch param_name

    case 'opt'
        fds = fieldnames(Solver.(sprintf('DEFAULT_OPT',param_name)));
    otherwise
        fds = fieldnames(DEFAULT.(sprintf('%s',param_name)));
    end

    for i = 1 : length(fds)
    	params.(fds{i}) = p.Results.(fds{i});
    end
end

