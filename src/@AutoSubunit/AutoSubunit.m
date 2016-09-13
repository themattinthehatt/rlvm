classdef AutoSubunit
    
% Class implementing the autoencoder subunit of an RLVM object
%   (Notes)
%
% Reference:
%
% Author: Matt Whiteway
%   06/30/16

properties
    w1              % matrix of weights from input to hidden layer
    w2              % matrix of weights from hidden to output layer
    b1              % vector of bias weights from input to hidden layer
    b2              % vector of bias weights from hidden to output layer
    num_cells       % number of input/output nodes
    num_hid_nodes   % number of hidden nodes in autoencoder network
    act_func_hid    % string specifying hidden layer activation function
    weight_tie      % boolean
    reg_lambdas     % struct defining regularization hyperparameters
    init_params     % struct saving initialization parameters
      % rng_state
      % auto_init_filt
    latent_vars    % matrix of latent variable states when inferring them 
                   % (after hidden unit activation function)
end

properties (Hidden) % inherited from RLVM
	allowed_auto_regtypes = {'l2_biases', 'l2_biases1', 'l2_biases',...
                             'l2_weights', 'l2_weights1', 'l2_weights2',...
                             'l1_hid' ,'d2t_hid', 'l1_hid_param'};
    allowed_auto_NLtypes = {'lin', 'relu', 'sigmoid', 'softplus'};
    min_pred_rate
    max_g = 50
end

%% ********************  constructor **************************************
methods
      
    function subunit = AutoSubunit(num_cells, num_hid_nodes, init_filt, ...
                                   act_func_hid, weight_tie)
    % subunit = AutoSubunit(num_cells, num_hid_nodes, init_filt, ...
    %                                    <act_func_hid>, <weight_tie>);
    % 
    % Constructor for AutoSubunit class
    %
    % INPUTS:
    %   num_cells:
    %   num_hid_nodes:
    %   init_filt:      string specifying a random initialization 
    %                   ('gaussian' or 'uniform') or a vector of the
    %                   appropriate size for the intended network structure
    %   <act_func_hid>: optional; string specifying activation function for
    %                   hidden layer; see allowed_auto_acttypes for 
    %                   supported funcs
    %   <weight_tie>:   optional; boolean indicating whether encoding 
    %                   weights should be the same (1) or different (0)
    %
    % OUTPUTS:
    %   subunit: initialized AutoSubunit object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return
    end
    
    % define defaults
    if (nargin < 5 || isempty(weight_tie))
        weight_tie = 1;
    end
    if (nargin < 4 || isempty(act_func_hid))
        act_func_hid = 'relu';
    end
    
    % parse inputs
    assert(num_cells > 0, ...
        'Must have positive number of cells')
    assert(num_hid_nodes > 0, ...
        'Must have positive number of hidden nodes')
    assert(ismember(act_func_hid, subunit.allowed_auto_NLtypes), ...
        'Invalid hidden layer activation function')
    assert(ismember(weight_tie, [0, 1]), ...
        'Invalid weight_tie parameter')
    
    % initialize weights
    [W1, W2, B1, B2, init_params_] = AutoSubunit.set_init_weights_stat(...
        init_filt, num_cells, num_hid_nodes, weight_tie);
        
    % set properties
    subunit.w1 = W1;
    subunit.w2 = W2;
    subunit.b1 = B1;
    subunit.b2 = B2;
    subunit.num_cells = num_cells;
    subunit.num_hid_nodes = num_hid_nodes;
    subunit.act_func_hid = act_func_hid;
    subunit.weight_tie = weight_tie;
    subunit.reg_lambdas = AutoSubunit.init_reg_lambdas(); % init all to 0s
    subunit.init_params = init_params_;
    subunit.latent_vars = [];
    
    end

end
%% ********************  setting methods **********************************
methods
    
    function subunit = set_params(subunit, varargin)
    % subunit = subunit.set_params(kargs)
    %
    % Takes a sequence of key-value pairs to set network parameters for an 
    % AutoSubunit object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'num_cells', scalar
    %           number of cells
    %       'num_hid_nodes', scalar
    %           number of hidden layer nodes
    %       'act_func_hid', string
    %           'lin' | 'relu' | 'sigmoid' | 'softplus'
    %           specify hidden layer activation functions
    %       'weight_tie', boolean
    %           1 to force encoding and decoding weights to be the same, 0 
    %           otherwise
    %   
    % OUTPUTS:
    %   subunit: updated AutoSubunit object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'num_cells'
                assert(varargin{i+1} > 0, ...
                    'must provide positive number of cells')
                if varargin{i+1} ~= subunit.num_cells
                    warning('changing number of cells; randomly reinitializing weights!')
                    subunit.num_cells = varargin{i+1};
                    subunit = subunit.set_init_weights('uniform');
                end
            case 'num_hid_nodes'
                assert(varargin{i+1} > 0, ...
                    'must provide positive number of hidden nodes')
                if varargin{i+1} ~= subunit.num_hid_nodes
%                     warning('changing number of hidden nodes; randomly reinitializing weights!')
                    subunit.num_hid_nodes = varargin{i+1};
                    subunit = subunit.set_init_weights('uniform');
                end
            case 'act_func_hid'
                assert(ismember(varargin{i+1}, subunit.allowed_auto_NLtypes), ...
                    'invalid activation function');
                subunit.act_func_hid = varargin{i+1};
            case 'weight_tie'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''weight_tie'' input must be boolean');
                if varargin{i+1} < subunit.weight_tie
                    % introducing new weights
                    subunit.weight_tie = varargin{i+1};
                    fprintf('Removing weight-tying constraint! Enter:\n')
                    fprintf('\t0 to reinitialize ALL weights randomly\n')
                    fprintf('\t1 to copy current weights\n')
                    user_input = input('(0/1): ');
                    if user_input == 1
                        subunit.w1 = subunit.w2';
                    elseif user_input == 0
                        subunit = subunit.set_init_weights('uniform');
                    else
                        warning('invalid input; reinitializing all weights randomly')
                        subunit = subunit.set_init_weights('uniform');
                    end
                elseif varargin{i+1} > subunit.weight_tie
                    % introducing weight-tying
                    subunit.weight_tie = varargin{i+1};
                    fprintf('Introducing weight-tying constraint! Enter:\n')
                    fprintf('\t0 to reinitialize ALL weights randomly\n')
                    fprintf('\t1 to use current weights\n')
                    user_input = input('(0/1): ');
                    if user_input == 1
                        % no changes
                    elseif user_input == 0
                        subunit = subunit.set_init_weights('uniform');
                    else
                        warning('invalid input; reinitializing all weights randomly')
                        subunit = subunit.set_init_weights('uniform');
                    end
                else
                    warning('No change in weight_tie; ignoring input')
                end
            otherwise
                error('Invalid input flag')
        end
        i = i + 2;
    end
    
    end % method
    
    
    function subunit = set_reg_params(subunit, varargin)
    % subunit = subunit.set_reg_params(kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters 
    % for an AutoSubunit object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'reg_type', scalar
    %           reg_types:
    %           'l2_weights' | 'l2_weights1' | 'l2_weights2' | 'l2_biases'
    %           | 'l2_biases1' | 'l2_biases2' | 'l1_hid' | 'd2t_hid'
    %       'kl_param', scalar
    %           scalar in [0 1] specifying the KL sparsity penalty value if
    %           using 'l1_hid' with sigmoid activation functions
    %
    % OUTPUTS:
    %   subunit: updated AutoSubunit object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'l2_weights'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_weights1 = varargin{i+1};
				subunit.reg_lambdas.l2_weights2 = varargin{i+1};
			case 'l2_weights1'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_weights1 = varargin{i+1};
			case 'l2_weights2'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_weights2 = varargin{i+1};
            case 'l2_biases'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_biases1 = varargin{i+1};
				subunit.reg_lambdas.l2_biases2 = varargin{i+1};
			case 'l2_biases1'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_biases1 = varargin{i+1};
			case 'l2_biases2'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l2_biases2 = varargin{i+1};
            case 'l1_hid'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.l1_hid = varargin{i+1};
			case 'd2t_hid'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                subunit.reg_lambdas.d2t_hid = varargin{i+1};
            case 'kl_param'
                assert(varargin{i+1} >= 0 && varargin{i+1} <= 1, ...
                    'KL param must lie in interval [0, 1]')
                subunit.reg_lambdas.kl_param = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    end % method
    
    
    function subunit = set_hidden_order(subunit, pop_activity)
    % subunit = subunit.set_hidden_order(pop_activity)
    %
    % Reorders the hidden units from highest to lowest stddev of output on
    % a particular set of population activity data
    %
    % INPUTS:
    %   pop_activity:   T x num_cells matrix of data
    %   
    % OUTPUTS:
    %   subunit: updated AutoSubunit object
    
    % check inputs
    subunit.check_inputs(pop_activity);
    
    % get stddev of hidden units
    hidden_act = subunit.get_model_internals(pop_activity);
    hidden_act = std(hidden_act{1}, [], 1);
    % sort by stddev (and flip to go from highest to lowest)
    [~, sorted_indxs] = sort(hidden_act);
    sorted_indxs = fliplr(sorted_indxs);
    % rearrange weights and biases
    subunit.b1 = subunit.b1(sorted_indxs);
    subunit.w1 = subunit.w1(:,sorted_indxs);
    subunit.w2 = subunit.w2(sorted_indxs,:);
    
    end % method
    
    
    function subunit = set_init_weights(subunit, init_weights)
    % subunit = subunit.set_init_weights(init_weights)
    %
    % Sets w1, w2, b1 and b2 properties of AutoSubunit object
    %
    % INPUTS:
    %   init_weights:   either a string specifying type of random 
    %                   initialization for weights ('gaussian' or 
    %                   'uniform') or a weight vector of appropriate length  
    %
    % OUTPUT:
    %   subunit: updated AutoSubunit object
    
    % call static set_init_weights used in Constructor
    [W1,W2,B1,B2,init_params_struct] = AutoSubunit.set_init_weights_stat(...
                                          init_weights, ...
                                          subunit.num_cells, ...
                                          subunit.num_hid_nodes, ...
                                          subunit.weight_tie);
    % set properties
    subunit.w1 = W1;
    subunit.w2 = W2;
    subunit.b1 = B1;
    subunit.b2 = B2;
    subunit.init_params = init_params_struct;
    
    end % method
    
end
%% ********************  getting methods **********************************
methods
    
    function [fgint, gint] = get_model_internals(subunit, pop_activity, varargin)
    % subunit = subunit.get_model_internals(pop_activity, pop_activity, kargs)
    %
    % Retrieves informaton from autoencoder model
    %
    % INPUTS:
    %   pop_activity:       T x num_cells matrix of responses
    % 
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is to use all data)
    %
    % OUTPUTS:
    %   gint:   2x1 cell array, each cell containing a matrix of the signal
    %           before being passed through the activation function
    %   fgint:  cell array of signal values after being passed through the
    %           hidden layer activation function
    %
    % CALLED BY:
    %   AutoSubunit.get_reg_pen, AutoSubunit.set_hidden_order
    
    % check inputs
    subunit.check_inputs(pop_activity);

    % define defaults
    indx_tr = NaN; % NaN means we use all available data

    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(pop_activity,1))) ...
                || isnan(varargin{i+1}), ...
                'Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end

    % use indx_tr
    if ~isnan(indx_tr)
        pop_activity = pop_activity(indx_tr,:);
    end

    % set aside constants to keep things clean
    W1 = subunit.w1;
    W2 = subunit.w2;
    B1 = subunit.b1;
    B2 = subunit.b2;

    % evaluate model
    if isempty(subunit.latent_vars)
        % autoencoder model
        gint{1}  = bsxfun(@plus, pop_activity * W1, B1');
    else
        % "unhinged" model
        gint{1}  = subunit.latent_vars;
    end
    fgint{1} = subunit.apply_act_func(gint{1});
    gint{2}  = bsxfun(@plus, fgint{1} * W2, B2');
    
    end % method
    
    
    function reg_pen = get_reg_pen(subunit, pop_activity, varargin)
    % reg_pen = subunit.get_reg_pen(pop_activity, kargs)
    %
    % Gets regularization penalties on AutoSubunit model parameters
    %
    % INPUTS:
    %   pop_activity:       T x num_cells matrix of responses
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is to use all data)
    %
    % OUTPUTS:
    %   reg_pen: struct containing penalties due to different regs
    
    % check inputs
    subunit.check_inputs(pop_activity);
    
    % define defaults
    indx_tr = NaN; % NaN means we use all available data
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1},1:size(pop_activity,1))) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % use indx_tr
    if ~isnan(indx_tr)
        pop_activity = pop_activity(indx_tr,:);
    end
    
    % set aside constants to keep things clean
    W1 = subunit.w1;
    W2 = subunit.w2;
    B1 = subunit.b1;
    B2 = subunit.b2;
    lambda_w1 = subunit.reg_lambdas.l2_weights1;
    lambda_w2 = subunit.reg_lambdas.l2_weights2;
    lambda_b1 = subunit.reg_lambdas.l2_biases1;
    lambda_b2 = subunit.reg_lambdas.l2_biases2;
    lambda_l1_hid = subunit.reg_lambdas.l1_hid;
    lambda_d2t_hid = subunit.reg_lambdas.d2t_hid;
    T = size(pop_activity,1);
    
    % sparsity penalty eval on hidden layer
    hidden_act = subunit.get_model_internals(pop_activity);
    sparse_func = subunit.get_sparse_penalty(hidden_act{1});
	
    % smoothness penalty eval on hidden layer
    reg_mat = spdiags([ones(T, 1) -2*ones(T, 1) ones(T, 1)], [-1 0 1], T, T);
    smooth_func = 0.5*sum(sum((reg_mat*hidden_act{1}).^2)) / T;

    % get penalty terms
    reg_pen.l2_weights1 = 0.5*lambda_w1*sum(sum(W1.^2)); % L2 on layer 1 weights
    reg_pen.l2_weights2 = 0.5*lambda_w2*sum(sum(W2.^2)); % L2 on layer 2 weights
    reg_pen.l2_biases1 = 0.5*lambda_b1*sum(B1.^2);       % L2 on layer 1 biases
    reg_pen.l2_biases2 = 0.5*lambda_b2*sum(B2.^2);       % L2 on layer 2 biases
    reg_pen.l1_hid = lambda_l1_hid*sparse_func;          % sparsity pen   
    reg_pen.d2t_hid = lambda_d2t_hid*smooth_func;        % smoothness pen
    
    end % method

    
    function [func, grad] = get_sparse_penalty(subunit, act)
    % [func, grad] = subunit.get_sparse_penalty(a2)
    %
    % Evaluates sparsity term in the cost function and its gradient; 
    % enforces sparsity in activity of hidden layer
    %
    % INPUTS:
    %   act:    T x num_hid_nodes matrix of activation in the hidden layer 
    %
    % OUTPUTS:
    %   func:   function value (scalar)
    %   grad:   gradient (column vector)
    
    switch subunit.act_func_hid
        case 'lin'
            func = sum(abs(act(:))) / size(act,1);
            grad = sign(sum(abs(act),1))' / size(act,1);
        case 'relu'
            func = sum(act(:)) / size(act,1);
            grad = sign(sum(abs(act),1))' / size(act,1);
        case 'sigmoid'
            p = subunit.reg_lambdas.kl_param;
            pj = sum(act,1);
            func = sum(p*log(p./pj)+(1-p)*log((1-p)./(1-pj)))/size(act,1);
            grad = (-p./pj+(1-p)./(1-pj))'/size(act,1);
        otherwise
            error('Sparsity penalty unsupported for current act_func_hid')
    end
    
    end % method
    
    
    function [w1, w2, b1, b2] = get_weights(subunit, weight_vec)
    % [w1, w2, b1, b2] = subunit.get_weights(weight_vec)
    %
    % Takes a column vector of the network parameters (weights and biases) 
    % and returns the separated parameters in matrix and vector form, 
    % respectively. No error checking here since this function is called
    % many times in the optimization routine.
    %
    % INPUTS:
    %   weight_vec:   assumed to be of the form:
    %                 [-----w1-----|------w2------|--b1--|--b2--]'
    % OUTPUTS:
    %   w1:           num_cells x num_hid_nodes matrix of weights
    %   w2:           num_hid_nodes x num_cells matrix of weights
    %   b1:           num_hid_nodes x 1 vector of hidden layer biases
    %   b2:           num_cells x 1 vector of output layer biases

    if subunit.weight_tie
        % no w1 matrix; just [---w1----|-b1-|-b2-]
        % get cut points
        cut1 = subunit.num_cells*subunit.num_hid_nodes;
        cut2 = cut1+subunit.num_hid_nodes;
        % get weights
        w2 = weight_vec(1:cut1);
        w2 = reshape(w2,subunit.num_hid_nodes,subunit.num_cells);
        w1 = w2';
        % get biases
        b1 = weight_vec(cut1+1:cut2);
        b2 = weight_vec((cut2+1):(cut2+subunit.num_cells));
    else
        % get cut points
        cut1 = subunit.num_cells*subunit.num_hid_nodes;
        cut2 = 2*cut1;
        cut3 = cut2+subunit.num_hid_nodes;
        % get weights
        w1 = weight_vec(1:cut1);
        w1 = reshape(w1,subunit.num_cells,subunit.num_hid_nodes);
        w2 = weight_vec(cut1+1:cut2);
        w2 = reshape(w2,subunit.num_hid_nodes,subunit.num_cells);
        % get biases
        b1 = weight_vec(cut2+1:cut3);
        b2 = weight_vec((cut3+1):(cut3+subunit.num_cells));
    end
    
    end % method
    
    
    function [w2, b2] = get_decoding_weights(subunit, weight_vec)
    % [w2, b2] = subunit.get_decoding_weights(weight_vec)
    %
    % Takes a column vector of the network parameters (weights and biases) 
    % and returns the separated parameters in matrix and vector form, 
    % respectively. No error checking here since this function is called
    % many times in the optimization routine.
    %
    % INPUTS:
    %   weight_vec:   assumed to be of the form:
    %                 [------w2------|--b2--]'
    % OUTPUTS:
    %   w2:           num_hid_nodes x num_cells matrix of weights
    %   b2:           num_cells x 1 vector of output layer biases

    % get cut point
    cut = subunit.num_cells * subunit.num_hid_nodes;
    % get weights
    w2 = reshape(weight_vec(1:cut), subunit.num_hid_nodes, []);
    % get biases
    b2 = weight_vec((cut+1):end);
    
    end % method
    
end
%% ********************  display methods **********************************
methods
    
    function fig_handle = disp_weights(subunit)
    % subunit.display_weights()
    %
    % Plots weights from hidden layer to output layer of AutoSubunit object
    %
    % INPUTS:
    %
    % OUTPUTS:
    %   fig_handle: figure handle of output
    
    fig_handle = figure;
    imagesc(subunit.w2',[-max(abs(subunit.w2(:))),max(abs(subunit.w2(:)))]);
    colormap(jet);
    
    end % method
    
    
    function [sorted_weights, sorted_indxs] = sort_weights(subunit, varargin)
    % [sorted_weights, sorted_indxs] = subunit.sort_weights(<method>)
    %
    % Sorts decoding matrix for nice presentation
    %
    % INPUTS:
    %	method:     optional;
    %               1 - hierarchical clustering
    %               2 - distance
    %               [3] - thresholding
    %               4 - kmeans
    %
    % OUTPUTS:
    %	sorted_clusters: num_cells x num_hid_nodes decoding matrix
    %	sorted_indxs:    vector mapping original to sorted indices

    % set defaults
    method = 3;
    
    % parse inputs
    if ~isempty(varargin)
        assert(ismember(varargin{1}, 1:4), ...
            'method must be 1, 2, 3 or 4')
        method = varargin{1};
    end
    
    % define constants
    num_clust = subunit.num_hid_nodes;
    clusters = subunit.w2';
    sorted_indxs = (1:subunit.num_cells)';

    if method == 1

        Z = linkage(pdist(clusters));
        [~, T, outperm] = dendrogram(Z);

        count = 1;
        for i = 1:length(outperm)
            indxs = find(T == outperm(i));
            indxs = indxs';
            % update indices
            sorted_indxs(count:(count+length(indxs)-1)) = indxs;
            % update count
            count = count + length(indxs);
        end

    elseif method == 2

        P = NaN(subunit.num_cells);
        for i = 1:subunit.num_cells
            % Euclidean distance
            for j = (i+1):subunit.num_cells
                P(i,j) = (clusters(i,:) - clusters(j,:)) * ...
                         (clusters(i,:) - clusters(j,:))';
                P(j,i) = P(i,j);
            end
        end

        % find cell that isn't part of any cluster
        [~,min_col] = min(sum(abs(clusters),2));
        P(1:(subunit.num_cells+1):subunit.num_cells*subunit.num_cells) = Inf;
        sorted_indxs(1) = min_col;

        % loop through other neurons, using most similar neuron as next 
        prev_cell = min_col;
        for i = 2:subunit.num_cells

            % find cell that is closest to previous
            [~,indx] = min(P(:,prev_cell));
            sorted_indxs(i) = indx;
            P(prev_cell,:) = Inf(1,subunit.num_cells);
            prev_cell = indx;

        end

    elseif method == 3

        thresh = 0.25;
        thresh_indx = 1;
        sorted_weights = clusters;
        for i = 1:num_clust
            if i ~= 1
                thresh = 0.1;
            end
            % sort ith cluster
            vals_to_sort = sorted_weights(thresh_indx:end,i);
            [vals_to_sort,indxs] = sort(vals_to_sort, 1, 'descend');
            %
            shifted_indxs = indxs + thresh_indx - 1;
            sorted_weights(thresh_indx:end,:) = sorted_weights(shifted_indxs,:);
            sorted_indxs(thresh_indx:end) = sorted_indxs(shifted_indxs);
            % find new threshold
            if i ~= num_clust
                [~,new_thresh_indx] = min(abs(vals_to_sort-thresh*max(vals_to_sort)));
                thresh_indx = thresh_indx - 1 + new_thresh_indx;
            end
        end
        sorted_indxs = flipud(sorted_indxs);

    elseif method == 4

        id = kmeans(clusters,num_clust);

        P = NaN(subunit.num_cells);
        for i = 1:subunit.num_cells
            % Euclidean distance
            for j = (i+1):subunit.num_cells
                P(i,j) = (clusters(i,:) - clusters(j,:)) * ...
                         (clusters(i,:) - clusters(j,:))';
                P(j,i) = P(i,j);
            end
        end

        % find cell that isn't part of any cluster
        P(1:(subunit.num_cells+1):subunit.num_cells*subunit.num_cells) = Inf;

        % loop through other neurons, using most similar neuron as next 
        count = 1;
        for i = 1:num_clust

            dist_to_mean = bsxfun(@minus,clusters,mean(clusters(id==i,:)));
            [~,min_col] = min(sum(dist_to_mean.^2,2));
            id(min_col) = 0;
            sorted_indxs(count) = min_col;
            count = count + 1;
            prev_cell = min_col;

            clust_size = length(find(id==i));
            Ptemp = P;
            Ptemp(setdiff(1:subunit.num_cells,find(id==i)),:) = Inf;

            for j = 1:clust_size
                % find cell that is closest to previous WITHIN the same cluster
                [~,indx] = min(Ptemp(:,prev_cell));
                sorted_indxs(count) = indx;
                Ptemp(prev_cell,:) = Inf(1,subunit.num_cells);
                P(prev_cell,:) = Inf(1,subunit.num_cells);
                prev_cell = indx;
                count = count + 1;
            end
        end
    end

    sorted_weights = clusters(sorted_indxs,:);

    % invert so that strongest weights are at top
    sorted_weights = flipud(sorted_weights);
    sorted_indxs = flipud(sorted_indxs);
    
    end % method
    
end
%% ********************  hidden methods ***********************************
methods (Hidden)
    
    function sig = apply_act_func(subunit, sig)
    % sig = subunit.apply_act_func(sig)
    %
    % internal function that applies activation function of the nodes
    % in the hidden layer to a given input
    %
    % INPUTS:
    %   sig:    T x num_hid_nodes matrix
    %
    % OUTPUTS:
    %   sig:    input passed through activation function

    switch subunit.act_func_hid
        case 'lin'
        case 'relu'
            sig = max(0, sig);
        case 'sigmoid'
            sig = 1 ./ (1 + exp(-sig));
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of under/overflow
            temp_sig(sig > subunit.max_g) = sig(sig > subunit.max_g);	% appx linear
            temp_sig(temp_sig < subunit.min_pred_rate) = subunit.min_pred_rate;	% so LL isn't undefined
            sig = temp_sig;
    end
    
    end % method

    
    function sig = apply_act_deriv(subunit, sig)
    % sig = subunit.apply_act_deriv(sig)
    %
    % internal function that calculates the derivative of the activation 
    % function of the nodes in the hidden layer to a given input
    %
    % INPUTS:
    %   sig:      T x num_hid_nodes matrix
    %
    % OUTPUTS:
    %   sig:      input passed through derivative of activation function

    switch subunit.act_func_hid
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            sig(sig <= 0) = 0;
            sig(sig > 0) = 1;
        case 'sigmoid'
            sig = exp(-sig) ./ (1 + exp(-sig)).^2;
        case 'softplus'
            temp_sig = exp(sig) ./ (1 + exp(sig));
            temp_sig(sig > subunit.max_g) = 1; % e^x/(1+e^x) => 1 for large x
            sig = temp_sig;
    end
    
    end % method
   
    
    function check_inputs(subunit, pop_activity)
    % subunit.check_inputs(pop_activity)
    %
    % Checks if the parameters of subunit are consistent with the given
    % population activity; used in any method that uses pop_activity as an
    % input.
    %
    % INPUTS:
    %   pop_activity:   T x num_cells matrix of responses
    %
    % OUTPUTS:
    %   none; throws error flag if parameters are not consistent
    %
    % CALLED BY:
    %   AutoSubunit.get_model_internals, AutoSubunit.get_reg_pen,
    %   AutoSubunit.get_hidden_order
    
    assert(size(pop_activity, 2) == size(subunit.w2, 2), ...
        'pop_activity inconsistent with autoencoder weights')

    end
    
end
%% ********************  static methods ***********************************
methods (Static)
    
    function reg_lambdas = init_reg_lambdas()
    % reg_lambdas = AutoSubunit.check_inputs(pop_activity)
    %
    % creates reg_lambdas struct and sets default values to 0; called from
    % AutoSubunit constructor
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_lambdas: struct containing initialized reg params
    
    reg_lambdas.l2_biases1 = 0;     % L2 on bias params
	reg_lambdas.l2_biases2 = 0;     % L2 on bias params
    reg_lambdas.l2_weights1 = 0;    % L2 on weights
	reg_lambdas.l2_weights2 = 0;    % L2 on weights
    reg_lambdas.l1_hid = 0;			% L1 on hidden layer activation
	reg_lambdas.d2t_hid = 0;		% d2t on hidden layer activation
    reg_lambdas.kl_param = 0;	    % sparsity param on hid layer act
    
    end % method
 
    
    function [w1, w2, b1, b2, init_params] = set_init_weights_stat( ...
                init_weights, num_cells, num_hid_nodes, weight_tie)
                                            
    % [w1, w2, b1, b2, init_params] = ...
    %       AutoSubunit.set_init_weights( init_weights, num_cells, ...
    %                              num_hid_nodes, weight_tie)
    % 
    % static function that initializes weights and sets init_params
    % structure based on input. Called from the AutoSubunit constructor 
    % and from the non-static method set_init_weights
    %
    % INPUTS:
    %   init_weights:   either a string specifying type of random 
    %                   initialization for weights ('gaussian' or 
    %                   'uniform') or a weight vector of appropriate length
    %   num_cells:      number of cells in model
    %   num_hid_nodes:  number of hidden nodes in model
    %   weight_tie:     0 or 1 specifying whether or not encoding and
    %                   decoding weights are the same (1) or different (0)
    % 
    % OUTPUTS:
    %   w1:             num_io_nodes x num_hid_nodes weight matrix
    %   w2:             num_hid_nodes x num_io_nodes weight matrix
    %   b1:             num_hid_nodes x 1 bias vector for hidden layer
    %   b2:             num_io_nodes x 1 bias vector for output layer
    %   init_params:    struct specifying init_weights and rng_state, if
    %                   applicable
    
    if ischar(init_weights)
        % randomly initialize weights; start biases off at 0
        init_params.rng_state = rng();
        init_params.init_weights = lower(init_weights);
        % create filter
        s = 0.5;
        switch lower(init_weights)
            case 'gaussian'
                w1 = s * randn(num_cells, num_hid_nodes);
                if weight_tie
                    w2 = w1';
                else
                    w2 = s * randn(num_hid_nodes, num_cells);
                end
            case 'uniform'
                % we'll choose weights uniformly from the interval [-r, r]
                r = 4 * sqrt(6) / sqrt(num_hid_nodes + num_cells + 1);   
                w1 = rand(num_cells, num_hid_nodes) * 2 * r - r;
                if weight_tie
                    w2 = w1';
                else
                    w2 = rand(num_hid_nodes, num_cells) * 2 * r - r;                
                end
            otherwise
                error('incorrect initialization string')
        end
        b1 = zeros(num_hid_nodes, 1);
        b2 = zeros(num_cells, 1);
    elseif isvector(init_weights)
        % use 'init_weights' to initialize weights
        % split up initial vector into appropriate matrices and vectors
        if weight_tie 
            % no w1 matrix; just [---w1----|-b1-|-b2-]
            assert(length(init_weights) == ...
                (num_hid_nodes * num_cells + num_hid_nodes + num_cells), ...
                'init_weights vector has improper size')
            % get cut points
            cut1 = num_hid_nodes * num_cells;
            cut2 = cut1 + num_hid_nodes;
            % get weights
            w1 = init_weights(1:cut1);
            w1 = reshape(w1, num_cells, num_hid_nodes);
            w2 = w1';
            % get biases
            b1 = init_weights((cut1 + 1):cut2);
            b2 = init_weights((cut2 + 1):end);
        else
            assert(length(init_weights) == ...
                (2 * num_hid_nodes * num_cells + num_hid_nodes + num_cells), ...
                'init_weights vector has improper size');
            % get cut points
            cut1 = num_hid_nodes * num_cells;
            cut2 = 2 * cut1;
            cut3 = cut2 + num_hid_nodes;
            % get weights
            w1 = init_weights(1:cut1);
            w1 = reshape(w1, num_cells, num_hid_nodes);
            w2 = init_weights(cut1 + 1:cut2);
            w2 = reshape(w2, num_hid_nodes, num_cells);
            % get biases
            b1 = init_weights((cut2 + 1):cut3);
            b2 = init_weights((cut3 + 1):end);
        end
        init_params.rng_state = NaN;
    else
        warning('init_weights must be a string or a vector')
    end
    
    end % method

end

end



