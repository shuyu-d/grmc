function data = load_data(name_dataset, varargin)
    % 
    root_data = 'datasets/';
    opts = parseArgs('realdata', varargin{:});
    
    switch name_dataset 
        case 'ML100k'
            raw = load(sprintf('%s/ml-100k/u.data', root_data));
            spMat = sparse( raw(:,1), raw(:,2), raw(:,3), ... 
                            max(raw(:,1)), max(raw(:,2)) ) ; 
            % Otr = [raw(1:79999, 1), raw(1:79999, 2)]; 
            % Ot = [raw(80000:end, 1), raw(80000:end, 2)]; 
        otherwise
            error('The data set %s is not known to the load_data function yet...', dataset_name ); 
    end
    % data = struct('mat', spMat, 'dims', size(spMat),...
    %               'Omega_t', Ot, 'Omega_tr',Otr); 
    data = struct('mat', spMat, 'dims', size(spMat)); 
end

