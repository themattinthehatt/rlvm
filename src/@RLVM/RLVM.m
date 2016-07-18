classdef RLVM
    
% Object-oriented implementation of an autoencoder with a single hidden 
% layer that also incorporates a basic stimulus processing model 
%
% Reference:
%   NIM
%   Andrew Ng
%
% Author: Matt Whiteway
%   06/30/16

properties
    num_cells
    noise_dist
    spk_NL
    fit_params          % struct defining the fitting parameters
        % fit_auto
        % fit_stim_individual
        % fit_stim_shared
        % fit_overall_offsets
        % deriv_check
    optim_params
        % opt_routine
        % batch_size
        % max_iter
        % display
        % monitor
    stim_subunits       % array of stimulus subunit objects
        % filt
        % NL_type
        % mod_sign
        % x_target
        % reg_lambdas
        % stim_params
        % init_params
        %   rng_state
        %   stim_init_filt
    auto_subunit        % autoencoder subunit object
        % w1; w2; b1; b2
        % num_cells
        % num_hid_nodes
        % act_func_hid
        % weight_tie
        % reg_lambdas
        % init_params
        %   rng_state
        %   auto_init_filt
    fit_history         % struct array that saves result of each fitting step
        % fit_type
        % r2
        % first_order_opt
        % exit_msg
    offsets
    lambda_off
end

properties (Hidden)
    version = '1.0';    % version number
    date = date;        % date of fit
    allowed_noise_dists   = {'gauss', 'poiss'};
    allowed_spk_NLs       = {'lin', 'relu', 'sigmoid', 'softplus'};
    allowed_auto_regtypes = {'l2_biases', 'l2_biases1', 'l2_biases',...
                             'l2_weights', 'l2_weights1', 'l2_weights2',...
                             'l1_hid' ,'d2t_hid', 'l1_hid_param'};
    allowed_auto_NLtypes  = {'lin', 'relu', 'sigmoid', 'softplus'};
    allowed_stim_regtypes = {'l2', 'd2t', 'd2x', 'd2xt'};    
    allowed_stim_NLtypes  = {'lin', 'relu', 'softplus'};
    allowed_optimizers    = {'fminunc', 'minFunc', 'con'};
    min_pred_rate         = 1e-5;   % min val for logs
    max_g                 = 50;     % max val for exponentials
end

% methods that are defined in separate files
methods
    net = fit_weights_latent_vars(net, fit_struct);
    [net, weights, latent_vars] = fit_weights(net, fit_struct);
    [net, weights, latent_vars] = fit_latent_vars(net, fit_struct);
end


%% ********************* constructor **************************************
methods
      
    function net = RLVM(init_params, varargin)
    % net = RLVM(init_params, kargs) 
    %
    % constructor function for an RLVM object; sets
    % properties, including calling AutoSubunit and StimSubunit 
    % constructors (no regularization is set)
    %
    % INPUTS:
    %   init_params:        struct with the following fields
    %       stim_dims       struct array with each entry 
    %                       in form [num_lags, num_xpix, num_ypix] or [] if
    %                       no stimulus model is desired 
    %       num_cells       number of cells
    %       num_hid_nodes   number of hidden nodes in autoencoder; [] if no
    %                       autoencoder model is desired
    %
    %   optional key-value pairs: [defaults]
	%       'noise_dist', string
    %           ['gauss'] | 'poiss'
    %           specifies noise distribution for cost function
    %       'spkNL', string
    %           ['lin'] | 'relu' | 'sigmoid' | 'softplus'
	%       'fit_overall_offsets', boolean
    %           [0] | 1
    %           specifies whether to fit overall offsets (1) or not (0)
    %
    %       'stim_init_filt', vector or string
    %           ['gaussian'] | 'uniform' | vector
    %           a cell array where each entry is either a string specifying
    %           a random initialization ('gaussian' or 'uniform') or a 
    %           vector of the appropriate size for the intended x_target 
    %           [filt_coeffs x 1]
    %       'mod_signs', +/-1 or vector of +/-1s
    %           [1] | -1
    %           vector of weights that define subunits as exc (+1) or inh 
    %           (-1); if all subunits share same mod_sign, use single 
    %           scalar
    %       'NL_types', string or cell array of srings
    %           ['lin'] | 'relu' | 'softplus'
    %           cell array of nonlinearities for stimulus model, one entry
    %           for each subunit; if all subunits share same NL_type, use
    %           single string
    %       'x_targets', scalar or vector of scalars
    %           [1]
    %           vector of scalars that assign stimulus matrix that each 
    %           subunit will use for fitting; if all subunits share same 
    %           x_target, use single scalar
    %
    %       'auto_init_filt', string or vector
    %           ['uniform'] | 'gaussian' | vector of weights
    %           either a string specifying a random random initialization 
    %           ('gaussian' or 'uniform') or a vector of the appropriate 
    %           size [w1(:); w2(:); b1; b2]
    %       'weight_tie', boolean
    %           [1] | 0
    %           1 for weight-tying in autoencoder model, which enforces 
    %           encoding and decoding weights to be the same; 0 otherwise
    %       'act_func_hid', string
    %           ['relu'] | 'lin' | 'sigmoid' | 'softplus'
    %           activiation function for hidden layer in autoencoder
    %
    % OUTPUT:
    %   net: initialized RLVM object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return 
    end
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')

    % CHECK INIT_PARAMS
    % no error checking for now
    num_stims = length(init_params.stim_params);
    num_subunits = init_params.num_subunits;

    % -------------------- DEFINE DEFAULTS --------------------------------
    
    noise_distribution = 'gauss';
    spk_nonlinearity = 'lin';
    % fit_params
    if ~isempty(init_params.stim_params) 
        if init_params.stim_params(1).num_outputs > 1
            fit_stim_individual = 1;
            fit_stim_shared = 0;
        elseif init_params.stim_params(1).num_outputs == 1
            fit_stim_individual = 0;
            fit_stim_shared = 1;
        else
            error('num_outputs field of stim params must equal num_cells or 1')
        end
    else
        fit_stim_individual = 0;
        fit_stim_shared = 0;
    end
    if init_params.num_hid_nodes ~= 0     
        fit_auto = 1;
    else
        fit_auto = 0; 
    end 
    fit_overall_offsets = 0;

    % stim_subunit - default is gaussian noise, all subunits excitatory and
    % use first target
    stim_init_filts = repmat({'gaussian'}, 1, num_subunits);
    mod_signs = ones(1, num_subunits);                      
    NL_types = repmat({'lin'}, 1, num_subunits);             
    if num_subunits ~= num_stims
        x_targets = ones(1,num_subunits);   % same target for each subunit
    else
        x_targets = 1:num_subunits;         % one subunit per stimulus
    end

    % auto_subunit
    auto_init_filt = 'uniform'; % initialize with gaussian noise
    weight_tie = 1;             % weight tying
    act_func_hid = 'relu';      % relu default 

    % -------------------- PARSE INPUTS -----------------------------------
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            
            case 'noise_dist'
                assert(ismember(varargin{i+1}, net.allowed_noise_dists),...
                    'Invalid noise distribution')
                noise_distribution = varargin{i+1};
            case 'spk_nl'
                assert(ismember(varargin{i+1}, net.allowed_spk_NLs),...
                    'Invalid spiking nonlinearity')
                spk_nonlinearity = varargin{i+1};
            case 'fit_overall_offsets'
                assert(ismember(varargin{i+1}, [0,1]), ...
                    'fit_overall_offsets option must be set to 0 or 1')
                fit_overall_offsets = varargin{i+1};

            % stim_subunit
            case 'stim_init_filts'
                assert(iscell(varargin{i+1}), ...
                    '''stim_init_filt'' must be a cell array of strings or vectors')
                if all(cellfun(@(x) ischar(x), varargin{i+1}))
                    assert(all(ismember(varargin{i+1}, {'gaussian', 'uniform'})), ...
                        'Invalid random init option')
                    stim_init_filts = varargin{i+1};
                    % if stim_init_filts is specified as a single string, 
                    % default to using this stim_init_filts for all 
                    % subunits
                    if length(stim_init_filts) == 1 && num_subunits > 1
                        stim_init_filts = ...
                            repmat(stim_init_filts, 1, num_subunits); 
                    elseif length(stim_init_filts) ~= num_subunits
                        error('Invalid number of stim_init_filts')
                    end
                elseif all(cellfun('isreal', varargin{i+1}))
                    % still need to make sure dims match; do in StimSubunit
                    % construtor
                    stim_init_filts = varargin{i+1}; 
                else
                    error('''stim_init_filt'' must be a cell array of strings or vectors')
                end
            case 'mod_signs'
                assert(all(ismember(varargin{i+1},[-1,1])),...
                'Invalid mod_sign')
                mod_signs = varargin{i+1};
                % if mod_signs is specified as a single number, default to
                % using this mod_sign for all subunits
                if length(mod_signs) == 1 && num_subunits > 1
                    mod_signs = repmat(mod_signs,1,num_subunits); 
                elseif length(mod_signs) ~= num_subunits
                    error('Invalid number of mod_signs')
                end
            case 'nl_types'
                assert(all(ismember(varargin{i+1}, net.allowed_stim_NLtypes)), ...
                    'Invalid NL_type')
                NL_types = varargin{i+1};
                % if NL_types is specified as a single string, default to 
                % using this NL_type for all subunits
                if ~iscell(NL_types) && ischar(NL_types)
                    NL_types = cellstr(NL_types);
                end
                if length(NL_types) == 1 && num_subunits > 1
                    NL_types = repmat(NL_types,1,num_subunits);
                elseif length(NL_types) ~= num_subunits
                    error('Invalid number of NL_types')
                end
            case 'x_targets'
                assert(all(ismember(varargin{i+1}, 1:num_stims)),...
                    'Invalid x_target')
                x_targets = varargin{i+1};
                % if x_targets is specified as a single number, default to
                % using this x_target for all subunits
                if length(x_targets) == 1 && num_subunits > 1
                    x_targets = repmat(x_targets, 1, num_subunits); 
                elseif length(x_targets) ~= num_subunits
                    error('Invalid number of x_targets')
                end
                
            % auto_subunit
            case 'auto_init_filt'
                if ischar(varargin{i+1})
                    assert(ismember(varargin{i+1}, {'gaussian', 'uniform'}), ...
                        'Invalid random init option')
                    auto_init_filt = varargin{i+1};
                elseif isvector(varargin{i+1})
                    % still need to make sure dims match; do in AutoSubunit
                    % construtor
                    auto_init_filt = varargin{i+1}; 
                else
                    error('''auto_init_filt'' must be a string or a vector')
                end
            case 'weight_tie'
                assert(ismember(varargin{i+1}, [0,1]),...
                    'weight_tie option must be 0 or 1')
                weight_tie = varargin{i+1};
            case 'act_func_hid'
                assert(ismember(varargin{i+1}, net.allowed_auto_NLtypes), ...
                    'Invalid auto activation function')
                act_func_hid = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end

    % -------------------- SET PROPERTIES ---------------------------------
    
    % model params
    net.noise_dist = noise_distribution;
    net.spk_NL = spk_nonlinearity;
    net.num_cells = init_params.num_cells;
    % fit_params
    net.fit_params.fit_auto = fit_auto;
    net.fit_params.fit_stim_individual = fit_stim_individual;
    net.fit_params.fit_stim_shared = fit_stim_shared;
    net.fit_params.fit_overall_offsets = fit_overall_offsets;
    net.fit_params.deriv_check = 0;
    % optim_params defaults
    net.optim_params = RLVM.set_init_optim_params(); 
    % fit_history
    net.fit_history = struct([]);
    % offsets
    net.offsets = zeros(init_params.num_cells, 1);
    net.lambda_off = 0;

    % -------------------- INITIALIZE SUBUNITS ----------------------------
    
    % autoencoder subunit; initialized w/o regularization
    if net.fit_params.fit_auto
        net.auto_subunit = AutoSubunit(net.num_cells, ...
                                       init_params.num_hid_nodes, ...
                                       auto_init_filt, ... 
                                       act_func_hid, ...
                                       weight_tie);
        % inherit values
        net.auto_subunit.allowed_auto_regtypes = net.allowed_auto_regtypes;
        net.auto_subunit.allowed_auto_NLtypes = net.allowed_auto_NLtypes;
        net.auto_subunit.min_pred_rate = net.min_pred_rate;
        net.auto_subunit.max_g = net.max_g;       
    end
    
    % stimulus subunits; initialize w/o regularization, loop through
    if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
        % initialize empty array of StimSubunit objects
        stim_subunits_(num_subunits,1) = StimSubunit();
        for n = 1:num_subunits
            stim_subunits_(n) = StimSubunit( ...
                                    init_params.stim_params(x_targets(n)), ...
                                    stim_init_filts{n}, ...
                                    mod_signs(n), ...
                                    NL_types{n}, ...
                                    x_targets(n));
            % inherit values
            stim_subunits_(n).allowed_stim_NLtypes = ...
                net.allowed_stim_NLtypes;
            stim_subunits_(n).allowed_stim_regtypes = ...
                net.allowed_stim_regtypes;
            stim_subunits_(n).min_pred_rate = net.min_pred_rate;
            stim_subunits_(n).max_g = net.max_g;
        end
        net.stim_subunits = stim_subunits_;        
    end
    
    end % method

end

%% ********************* setting methods **********************************
methods
    
    function net = set_params(net, varargin)
    % net = net.set_params(kargs)
    %
    % Takes a sequence of key-value pairs to set network parameters for an 
    % RLVM object
    %
    % INPUTS:
    %   optional key-value pairs:
    %	'noise_dist', string
    %       'gauss' | 'poiss'
    %       specifies noise distribution used during learning of parameters
    %   'spk_NL', string
    %       ['lin'] | 'relu' | 'sigmoid' | 'softplus'
    %       specifies the spiking nonlinearity
    %   'num_cells', scalar
    %       number of cells
    % 
    % OUTPUTS:
    %   net: updated RLVM object

    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')

    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'noise_dist'
                assert(ismember(varargin{i+1}, net.allowed_noise_dists), ...
                    'Invalid noise distribution')
                net.noise_dist = varargin{i+1};
            case 'spk_nl'
                assert(ismember(varargin{i+1}, net.allowed_spk_NLs), ...
                    'Invalid spiking nonlineariy')
                net.spk_NL = varargin{i+1};
            case 'num_cells'
                assert(varargin{i+1} > 0, ...
                    'must provide positive number of cells')
                if varargin{i+1} ~= net.num_cells
                    warning('changing number of cells; randomly reinitializing weights!')
                    % update properties
                    net.num_cells = varargin{i+1};
                    net.auto_subunit.num_cells = varargin{i+1};
                    % update auto weights
                    if ~isempty(net.auto_subunit)
                        net.auto_subunit = ...
                            net.auto_subunit.set_init_weights('uniform');
                    end
                    % update stim weights
                    if ~isempty(net.stim_subunits)
                        for j = 1:length(net.stim_subunits)
                            net.stim_subunits(j) = ...
                                net.stim_subunits(j).set_init_filt('gaussian');
                        end
                    end
                end
            otherwise
            error('Invalid input flag')
        end
        i = i + 2;
    end
    
    end % method
    
    
    function net = set_fit_params(net, varargin)
    % net = net.set_fit_params(kargs)
    %
    % Takes a sequence of key-value pairs to set fit parameters for an 
    % RLVM object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'fit_auto', boolean
    %           specifies whether an autoencoder model will be fit or not
    %       'fit_stim_individual', boolean
    %           specifies whether individual stimulus subunits will be fit
    %           to each cell or not
    %       'fit_stim_shared', boolean
    %           specified whether shared stimulus subunits will be fit or
    %           not
    %       'fit_overall_offsets', boolean
    %           specifies whether overall offsets will be fit or not
    %       'deriv_check', boolean
    %           specifies numerical derivative checking
    %   
    % OUTPUTS:
    %   net: updated RLVM object

    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')

    % parse inputs
    i = 1; 
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'fit_auto'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''fit_auto'' option must be set to 0 or 1')
                net.fit_params.fit_auto = varargin{i+1};
            case 'fit_stim_individual'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''fit_stim_individual'' option must be set to 0 or 1')
                assert(~(varargin{i+1} == 1 ...
                    && net.fit_params.fit_stim_shared == 1), ...
                    'RLVM:invalidoption', ...
                    'cannot fit both individual and shared subunits')
                net.fit_params.fit_stim_individual = varargin{i+1};
            case 'fit_stim_shared'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''fit_stim_shared'' option must be set to 0 or 1')
                assert(~(varargin{i+1} == 1 ...
                    && net.fit_params.fit_stim_individual == 1), ...
                    'RLVM:invalidoption', ...
                    'cannot fit both individual and shared subunits')
                net.fit_params.fit_stim_shared = varargin{i+1};
            case 'fit_overall_offsets'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''fit_overall_offsets'' option must be set to 0 or 1')
                net.fit_params.fit_overall_offsets = varargin{i+1};
            case 'deriv_check'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    '''deriv_check'' option must be set to 0 or 1')
                net.fit_params.deriv_check = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    end % method
    
    
    function net = set_optim_params(net, varargin)
    % net = net.set_optim_params(kargs)
    %
    % Takes a sequence of key-value pairs to set optimization parameters 
    % for an RLVM object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'optimizer', string
    %           'minFunc' | 'fminunc' | 'minConf'       
    %           specify the optimization routine used for learning the
    %           weights and biases.
    %       'batch_size', scalar
    %           number of examples to use for each gradient descent step if
    %           using sgd
    %       'display', string
    %           'off' | 'iter' | 'batch'
    %           'off' to suppress output, 'iter' for output at each
    %           iteration, 'batch' for output after each pass through data 
    %           (in sgd)
    %       'max_iter', scalar
    %           maximum number of iterations of optimization routine
    %       'monitor', string
    %           'off' | 'iter' | 'batch' | 'both'
    %           'off' to suppress saving output, 'iter' to save output 
    %           at each iteration, 'batch' to save output after each pass 
    %           through data (in sgd), 'both' to save both
    %
    % OUTPUTS:
    %   net: updated Autoencoder object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    % allowed optimization routines
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'optimizer'
                assert(ismember(varargin{i+1}, net.allowed_optimizers), ...
                    'invalid optimizer')
                net.optim_params.optimizer = varargin{i+1};
            case 'batch_size'
                assert(mod(net.data_params.examples, varargin{i+1}) == 0, ...
                    'batch size should evenly divide number of examples')
                net.optim_params.batch_size = varargin{i+1};
            case 'display'
                assert(ismember(varargin{i+1}, {'off', 'iter', 'batch'}), ...
                    'invalid parameter for ''display''')
                net.optim_params.Display = varargin{i+1};
            case 'max_iter'
                assert(varargin{i+1} > 0, ...
                    'max number of iterations must be greater than zero')
                net.optim_params.max_iter = varargin{i+1};
                net.optim_params.maxIter = varargin{i+1};
                net.optim_params.maxFunEvals = 2*varargin{i+1};
            case 'monitor'
                assert(ismember(varargin{i+1}, {'off', 'iter', 'batch', 'both'}), ...
                    'invalid parameter for ''monitor''')
                net.optim_params.monitor = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    end % method
    
    
    function net = set_reg_params(net, reg_target, varargin)
    % net = net.set_reg_params(reg_target, kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters
    % for either the stimulus model or the autoencoder model in an RLVM
    % object. Note that both the StimSubunit class and the AutoSubunit 
    % class have an equivalent method; the main usefulness of this method 
    % is to quickly update the reg_params structure for ALL stimulus 
    % subunits.
    %
    % INPUTS:
    %   reg_target:     string specifying whether to apply the specified
    %                   reg params to the stim ('stim') or autoencoder 
    %                   ('auto') model, or to offsets ('off')
    %
    %   optional key-value pairs:
    %       'subs', vector
    %           specify set of subunits to apply the new reg_params
    %           to if applying reg_params to stim model (default = ALL)
    %       'lambda_type', scalar
    %           stim lambda_types:
    %           'l2' | 'd2t' | 'd2x' | 'd2xt'
    %           auto lambda_types:
    %           'l2_weights' | 'l2_weights1' | 'l2_weights2' | 'l2_biases'
    %           | 'l2_biases1' | 'l2_biases2' | 'l1_hid' | 'd2t_hid'
    %           first input is a string specifying the type of 
    %           regularization, followed by a scalar giving the associated 
    %           regularization value, which will be applied to the 
    %           autoencoder subunit or all stimulus subunits specified by 
    %           'subs'. If different lambda values are desired for
    %           different subunits, for now this method will have to be
    %           called for each subunit update
    %           Example: net = net.set_reg_params('auto', 'l2_weights', 10)
    %   
    % OUTPUTS:
    %   net: updated RLVM object
    
    % check for appropriate number of inputs
    assert(ismember(reg_target, {'stim', 'auto', 'off'}), ...
        'Must specify which subunit to update')
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    % parse inputs
    switch reg_target
        case 'stim'
            % look for sub_inds; if none specified, the reg_params will be 
            % applied to ALL subunits
            subs_loc = find(strcmp(varargin, 'subs'));
            if ~isempty(subs_loc)
                assert(all(ismember(varargin{subs_loc+1}, 1:length(net.stim_subunits))), ...
                    'invalid target subunits specified')
                sub_inds = varargin{subs_loc+1};
                % remove sub_inds from varargin; will be passed to another 
                % method
                varargin(subs_loc) = [];
                varargin(subs_loc) = []; % equiv to subs_loc+1 after delete 
            else
                % default is to update all subunits
                sub_inds = 1:length(net.stim_subunits); 
            end               
            % update reg_params for all desired stim subunits
            for i = 1:length(sub_inds)
                net.stim_subunits(sub_inds(i)) = ...
                    net.stim_subunits(sub_inds(i)).set_reg_params(varargin{:}); 
            end
        case 'auto'
            % update reg_params for auto subunit
            net.auto_subunit = net.auto_subunit.set_reg_params(varargin{:});
        case 'off'
            i = 1;
            while i <= length(varargin)
                switch lower(varargin{i})
                    case 'l2'
                        assert(varargin{i+1} >= 0, ...
                            'reg value must be nonnegative')
                        net.lambda_off = varargin{i+1};
                    otherwise
                        error('Invalid input flag');
                end
                i = i + 2;
            end
        otherwise
            error('Invalid reg_target; must be ''stim'',''auto'' or ''off''')
    end
    
    end % method
	
end
    
%% ********************* getting methods **********************************
methods

    function [r2, cost_func, mod_internals, mod_reg_pen] = ...
        get_model_eval(net, pop_activity, varargin)
    % [r2, cost_func, mod_internals, mod_reg_pen] = ...
    %		net.get_model_eval(pop_activity, <Xstims>, kargs)
    %
    % Evaluates current RLVM object and returns relevant model data
    % like goodness-of-fit (r2 or pseudo-r2), value of the cost function,
    % regularization information, etc.
    %
    % INPUTS:
    %   pop_activity:       T x num_cells matrix of responses
    %   Xstims:             optional; cell array of stimuli, each of which 
    %                       is T x _
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   r2:             num_cells x 1 vector of rsquared values 
    %   cost_fun:       scalar value of unregularized cost function
    %   auto_internals: struct with fields
    %       auto_gint   2x1 cell array, each cell containing a T x num_cells
    %                   matrix of the signal before being passed through
    %                   the activation function of hidden (1) and output
    %                   (2) layers
    %       auto_fgint  same as gint, except value of signal after being
    %                   passed through the activation function
    %       stim_gint   num_subunitsx1 cell array, each cell containing a
    %                   T x num_cells matrix of the filter outputs
    %       stim_fgint  num_subunitsx1 cell array, each cell containing a
    %                   T x num_cells matrix of the subunit outputs
    %   mod_reg_pen:        
    %       auto_reg_pen struct containing penalties due to different regs
    %       stim_reg_pen struct containing penalties due to different regs

    % check inputs
    % net = net.check_inputs(pop_activity,Xstims);

    % define defaults
    Xstims = [];
    indx_tr = NaN; % NaN means we use all available data
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        if ~ischar(varargin{i})
            % must be Xstims
            Xstims = varargin{i};
            i = i + 1;
        else
            switch lower(varargin{i})
                case 'indx_tr'
                    assert(all(ismember(varargin{i+1}, 1:size(pop_activity, 1))) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                    indx_tr = varargin{i+1};
                otherwise
                    error('Invalid input flag');
            end
            i = i + 2;
        end
    end

    % use indx_tr
    if ~isnan(indx_tr)
        pop_activity = pop_activity(indx_tr,:);
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:);
        end
    end
    
    T = size(pop_activity, 1);
    G = 0;
    
    % get internal generating signals - auto
    if ~isempty(net.auto_subunit)
        [auto_fgint, auto_gint] = ...
            net.auto_subunit.get_model_internals(pop_activity);
        G = G + auto_gint{2};      
    else
        auto_gint = [];
        auto_fgint = [];
    end
    % get internal generating signals - stim
    if ~isempty(net.stim_subunits)
        num_subunits = length(net.stim_subunits);
        if net.stim_subunits(1).stim_params.num_outputs == 1
            stim_gint = zeros(T, num_subunits);
            stim_fgint = zeros(T, num_subunits);
            for i = 1:num_subunits
                [stim_fgint(:,i), stim_gint(:,i)] = ...
                    net.stim_subunits(i).get_model_internals(Xstims);
            end
            G = G + stim_fgint{i} * net.stim_weights;
        elseif net.stim_subunits(1).stim_params.num_outputs == net.num_cells
            stim_gint = cell(num_subunits,1);
            stim_fgint = cell(num_subunits,1);
            for i = 1:num_subunits
                [stim_fgint{i}, stim_gint{i}] = ...
                    net.stim_subunits(i).get_model_internals(Xstims);
                G = G + stim_fgint{i} * net.stim_subunits(i).mod_sign;
            end
        end 
    else
        stim_gint = [];
        stim_fgint = [];
    end
    G = bsxfun(@plus, G, net.offsets');
    pred_activity = net.apply_spk_NL(G);
    
    % get regularization penalites - auto
    if ~isempty(net.auto_subunit)
        auto_reg_pen = net.auto_subunit.get_reg_pen(pop_activity);
    else
        auto_reg_pen = [];
    end
    % get regularization penalties - stim
    stim_reg_pen = [];
    for i = 1:length(net.stim_subunits)
        stim_reg_pen = ...
            cat(1, stim_reg_pen, net.stim_subunits(i).get_reg_pen());
    end

    % evaluate cost_func and pseudo-r^2
    switch net.noise_dist
        case 'gauss'
            Z = 2 * numel(pop_activity);
            LL = sum((pred_activity - pop_activity).^2, 1)';
        case 'poiss'
            Z = sum(sum((pop_activity)));
            LL = -sum(pop_activity.*log(pred_activity) - pred_activity, 1)';
        otherwise
            error('Invalid noise distribution')
    end
    LLnull = sum((bsxfun(@minus, pop_activity, mean(pop_activity, 1))).^2, 1)';
    r2 = 1 - LL./LLnull;
    cost_func = sum(LL) / Z;
    
    % evaluate reg_pen on offsets
    off_reg_pen = 0.5 * net.lambda_off * sum(net.offsets.^2);

    % put internal generating functions in a struct if desired in output
    if nargout > 2
        mod_internals.auto_gint = auto_gint;
        mod_internals.auto_fgint = auto_fgint;
        mod_internals.stim_gint = stim_gint;
        mod_internals.stim_fgint = stim_fgint;
        mod_internals.G = G;
    end
    % put reg penalties in a struct if desired in output
    if nargout > 3
        mod_reg_pen = struct('auto_reg_pen', auto_reg_pen, ...
                             'stim_reg_pen', stim_reg_pen, ...
                             'off_reg_pen', off_reg_pen);
    end

    end % method
  
end

%% ********************* fitting methods **********************************
methods
    
    function [net, varargout] = fit_model(net, fit_type, pop_activity, varargin)
    % net = net.fit_model(fit_type, pop_activity, <Xstims>, kargs)
    %
    % Checks inputs and farms out parameter fitting to other methods 
    % depending on what type of model fit is desired
    %
    % INPUTS:
    %   fit_type:           ['params'] | 'weights' | 'latent_vars' | 'alt'
    %   pop_activity:       T x num_cells matrix of responses
    %   Xstims:             optional; cell array of stimuli, each of which 
    %                       is T x _; required if stim_subunits is not
    %                       empty
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           fitting
    %       'subs', vector
    %           vector of indices specifying which subunits of stimulus 
    %           model should be optimized 
    %       'init_weights', string or vector
    %           ['model'] | 'gauss' | vector
    %           used by fit_weights, fit_latent_vars and fit_alt to 
    %           initialize model; specify 'model' to use current model 
    %           weights, 'gauss' to use a random initialization, or input
    %           a vector of the proper length (weights + biases)
    %       'init_latent_vars', string or vector
    %           ['model'] | 'gauss' | matrix
    %           used by fit_weights, fit_latent_vars and fit_alt to
    %           initialize model; specify 'model' to use latent vars
    %           generated by the current model, 'gauss' to use a random
    %           initialization, or input a T x num_hid_nodes matrix
    %       'init_type', string
    %           ['auto'] | 'rand' | 'vec'
    %           used by fit_alt to initialize model; 'auto' fits the
    %           autoencoder, 'rand' uses random values, and 'vec' uses the
    %           values stored in 'init_weights' and 'init_latent_vars' if
    %           they are vectors
    %       'first_fit', string
    %           ['weights'] | 'latent_vars'
    %           used by fit_alt to determine which set of parameters to fit
    %           first
    %
    % OUTPUTS:
    %   net: updated RLVM object
    
    assert(ismember(fit_type, {'params', 'weights', 'latent_vars', 'alt'}), ...
        'Unsupported fit_type specified')
    
    % DEFINE DEFAULTS
    % put all relevant data into fitting_struct, which acts as a common
    % data structure to all fitting routines
    Xstims = [];
    indx_tr = NaN;              % train on all data
    fit_subs = 1:length(net.stim_subunits);
    init_weights = 'model';
    init_latent_vars = 'model';
    init_type = 'auto';
    fit_first = 'latent_vars';
    
    % PARSE OPTIONAL INPUTS
    i = 1;
    while i <= length(varargin)
        if ~ischar(varargin{i})
            % this must be Xstims
            Xstims = varargin{i};
            i = i + 1;
        else
            switch lower(varargin{i})
                case 'indx_tr'
                    assert(all(ismember(varargin{i+1}, 1:size(pop_activity,1))) ...
                        && ~any(isnan(varargin{i+1})), ...
                        'Invalid fitting indices')
                    indx_tr = varargin{i+1};
                case 'subs'
                    assert(all(ismember(varargin{i+1}, 1:Nsubs)), ...
                        'invalid target subunits specified');
                    fit_subs = varargin{i+1};
                case 'init_weights'
                    if ischar(varargin{i+1})
                        assert(ismember(varargin{i+1}, {'model', 'gauss'}), ...
                            'Unsupported init_weights specified')
                    elseif isvector(varargin{i+1})
                        assert(length(varargin{i+1}) == ...
                            net.num_cells + net.num_cells * net.auto_subunit.num_hid_nodes, ...
                            'init_weights vector has improper size')
                    else
                        warning('Incorrect init_weights format; defaulting to model')
                        init_weights = 'model';
                    end
                    init_weights = varargin{i+1};
                case 'init_latent_vars'
                    if ischar(varargin{i+1})
                        assert(ismember(varargin{i+1}, {'model', 'gauss'}), ...
                            'Unsupported init_latent_vars specified')
                    elseif ismatrix(varargin{i+1})
                        if isnan(indx_tr)
                            assert(size(varargin{i+1}) == ...
                                [size(pop_activity, 1), net.auto_subunit.num_hid_nodes], ...
                                'Incorrect size for latent vars')
                        else
                            assert(size(varargin{i+1}) == ...
                                [length(indx_tr), net.auto_subunit.num_hid_nodes], ...
                                'Incorrect size for latent vars')
                        end
                    else
                        warning('Incorrect init_latent_vars format; defaulting to model')
                        init_latent_vars = 'model';
                    end
                    init_latent_vars = varargin{i+1};
                case 'init_type'
                    assert(ismemeber(varargin{i+1}, {'auto', 'rand', 'vec'}), ...
                        'Improper init_type specified')
                    init_type = varargin{i+1};
                case 'fit_first'
                    assert(ismember(varargin{i+1}, {'weights', 'latent_vars'}), ...
                        'Improper fit_first specified')
                    fit_first = varargin{i+1};
                otherwise
                    error('Invalid input flag');
            end
            i = i + 2;
        end
    end
    
    if isnan(indx_tr)
        fit_struct.pop_activity = pop_activity;
        fit_struct.Xstims = Xstims;
    else
        fit_struct.pop_activity = pop_activity(indx_tr,:);
        if ~isempty(Xstims)
            for i = 1:length(Xstims)
                fit_struct.Xstims{i} = Xstims{i}(indx_tr,:);
            end
        else
            fit_struct.Xstims = [];
        end
    end
    clear pop_activity
    clear Xstims
    fit_struct.fit_subs = fit_subs;
    fit_struct.init_weights = init_weights;
    fit_struct.init_latent_vars = init_latent_vars;
    fit_struct.init_type = init_type;
    fit_struct.fit_first = fit_first;
    
    % check consistency between inputs
    net.check_fit_struct(fit_struct, fit_type); 
    
    % DETERMINE OPTIMIZATION ROUTINE
    if strcmp(fit_type, 'params')
        net = net.fit_weights_latent_vars(fit_struct);
    elseif strcmp(fit_type, 'weights')
        [net, weights, latent_vars] = net.fit_weights(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    elseif strcmp(fit_type, 'latent_vars')
        [net, weights, latent_vars] = net.fit_latent_vars(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    elseif strcmp(fit_type, 'alt')
        [net, weights, latent_vars] = net.fit_alt(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    else
        error('Incorrect fit_type')
    end
    
    end % method
    
end

%% ********************* display methods **********************************
methods
    
    function fig_handle = disp_stim_filts(net, cell_num)
    % net = net.disp_stim_filts(cell_num)
    %
    % Plots stimulus filters of various subunits for a given cell
    %
    % INPUTS:
    %	cell_num:			cell number for plotting
    %
    % OUTPUTS:
    %   fig_handle:         handle of created figure

    num_subs = length(net.stim_subunits);

    % set figure position
    fig_handle = figure;
    set(fig_handle,'Units','normalized');
    set(fig_handle,'OuterPosition',[0.05 0.55 0.6 0.4]); %[left bottom width height]

    for i = 1:num_subs
        subplot(1,num_subs,i)
        net.stim_subunits(i).disp_filt(cell_num);
    end
    
    end % method
    
end

%% ********************* hidden methods ***********************************
methods (Hidden)
	
    function sig = apply_spk_NL(net, sig)
    % sig = net.apply_act_func(sig)
    %
    % internal function that applies spiking nonlinearity of model to input
    %
    % INPUTS:
    %   sig:    T x num_cells matrix
    %
    % OUTPUTS:
    %   sig:    input passed through spiking nonlinearity

    switch net.spk_NL
        case 'lin'
        case 'relu'
            sig = max(0, sig);
        case 'sigmoid'
            sig = 1 ./ (1 + exp(-sig));
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of under/overflow
            % appx linear
            temp_sig(sig > net.max_g) = sig(sig > net.max_g);
            % so LL isn't undefined
            temp_sig(temp_sig < net.min_pred_rate) = net.min_pred_rate;
            sig = temp_sig;
    end
    
    end % method

    
    function sig = apply_spk_NL_deriv(net, sig)
    % sig = net.apply_spk_NL_deriv(sig)
    %
    % internal function that calculates the derivative of the spiking 
    % nonlinearity to a given input
    %
    % INPUTS:
    %     sig:      T x num_cells matrix
    %
    % OUTPUTS:
    %     sig:      input passed through derivative of spiking nonlinearity

    switch net.spk_NL
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            sig(sig <= 0) = 0;
            sig(sig > 0) = 1;
        case 'sigmoid'
            sig = exp(-sig) ./ (1 + exp(-sig)).^2;
        case 'softplus'
            temp_sig = exp(sig) ./ (1 + exp(sig));
            temp_sig(sig > net.max_g) = 1; % e^x/(1+e^x) => 1 for large x
            sig = temp_sig;
    end
    
    end % method
    
    
    function net = check_inputs(net, pop_activity, Xstims)
    % net = net.check_inputs(pop_activity, Xstims)
    %
    % Checks inputs for fitting and eval methods
    %
    % INPUTS:
    %   pop_activity:       T x num_cells matrix of responses
    %   Xstims:             cell array of stimuli, each of which is Tx_
    %
    % OUTPUTS:
    %   none; throws flag if error
    %
    % CALLED BY:
    %   RLVM.get_model_eval
    
    % check pop_activity and Xstims
    if ~isempty(Xstims)
        % ensure Xstims is a cell array
        assert(iscell(Xstims), ...
            'Xstims must be a cell array')
        % ensure first dimension is same across all Xstims (if necessary)
        assert(length(unique(cellfun(@(x) size(x,1), Xstims))) == 1, ...
            'Xstim elements need to have same size along first dimension');
        % ensure first dimension is same across pop_activity and Xstims
        assert(size(pop_activity, 1) == size(Xstims{1}, 1), ...
            'First (time) dim must be consisent across pop activity and Xstims')
    end
    
    end % method
    
    
    function check_fit_struct(net, fit_struct, fit_type)
    % net.check_fit_struct(fit_struct)
    %
    % Checks input structure for fitting methods
    %
    % INPUTS:
    %   fit_struct: structure for parameter fitting; see fit_model method
    %               for relevant fields
    %
    % OUTPUTS:
    %   none; throws flag if error
    %
    % CALLED BY:
    %   RLVM.fit_model
    
    % check pop_activity and Xstims
    if ~isempty(fit_struct.Xstims)
        % ensure Xstims is a cell array
        assert(iscell(fit_struct.Xstims), ...
            'Xstims must be a cell array')
        % ensure first dimension is same across all Xstims (if necessary)
        assert(length(unique(cellfun(@(x) size(x,1), fit_struct.Xstims))) == 1, ...
             'Xstim elements need to have same size along first dimension');
        % ensure first dimension is same across pop_activity and Xstims
        assert(size(fit_struct.pop_activity, 1) == size(fit_struct.Xstims{1}, 1), ...
            'First (time) dim must be consisent across pop activity and Xstims')
    end
    
    if net.fit_params.fit_auto
    end
    if net.fit_params.fit_stim_individual
        % ensure Xstims exists
        assert(~isempty(fit_struct.Xstims), ...
            'must provide Xstims matrix to fit stim subunit')
    end
    if net.fit_params.fit_stim_shared
        % ensure Xstims exists
        assert(~isempty(fit_struct.Xstims), ...
            'must provide Xstims matrix to fit stim subunit')
    end
    
    if strcmp(fit_type, 'params')
    elseif strcmp(fit_type, 'weights')
    elseif strcmp(fit_type, 'latent_vars')
    end
   
    end % method
    
end

%% ********************* static methods ***********************************
methods (Static)
    
    function init_params = create_init_params(stim_params, num_cells, ...
                                    num_hid_nodes, varargin)
    % stim_params = RLVM.create_init_params(stim_params, num_cells,
    %                               <num_hid_nodes>, kargs)
    %
    % Creates a struct containing initial parameters to feed into
    % constructor
    %
    % INPUTS:
    %   stim_params:    struct array 
    %       dims:       defines dims of the (time-embedded) stimulus, in 
    %                   the form [num_lags, num_xpix, num_ypix]. If no
    %                   stimulus model is desired, use [].
    %   num_cells:      number of cells in the population
    %   num_hid_nodes:  optional; number of hidden units. If no autoencoder
    %                   is desired, use 0.
    %
    %   optional key-value pairs:
    %       'num_subs', scalar
    %           number of subunits for model
    %
    % OUTPUTS:
    %   init_params:    struct of initial parameters

    % check inputs
    if nargin < 3
        warning('no hidden nodes specified; defaulting to none')
        num_hid_nodes = 0;
    end
    if nargin < 2
        error('must input number of cells as second argument')
    end
    if nargin < 1
        error('must input stim_params struct array as first argument')
    end
    assert(num_cells > 0, ...
        'Must fit positive number of cells')
    assert(num_hid_nodes >= 0, ...
        'Must specify nonnegative number of hidden nodes')
    
    % make sure each dim field has form [num_lags num_xpix num_ypix] and
    % concatenate with 1's if necessary
    if ~isempty(stim_params)
        for n = 1:length(stim_params)
            while length(stim_params(n).dims) < 3
                % pad stim_dims with 1's for bookkeeping
                stim_params(n).dims = cat(2,stim_params(n).dims,1); 
            end
            % used in initializing weights
            stim_params(n).num_cells = num_cells; 
        end
    end
    
    % set defaults
    num_subunits = length(stim_params);
    
    % parse inputs
    i = 1; 
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'num_subs'
                assert(num_subs > 0, ...
                    'must have positive number of subunits')
                num_subunits = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % make sure there is at least as many subunits as stimuli
    assert(num_subunits >= length(stim_params), ...
        'Not enough subunits')
    
    % set fields
    init_params.stim_params = stim_params;
    init_params.num_subunits = num_subunits;
    init_params.num_cells = num_cells;
    init_params.num_hid_nodes = num_hid_nodes;

    end % method

    
    function optim_params = set_init_optim_params()
    % optim_params = RLVM.set_init_optim_params();
    %
    % Sets default optimization parameters for the various optimization 
    % routines

    % optim_params
    optim_params.optimizer   = 'minFunc'; % opt package to use
    optim_params.batch_size  = 10;        % batch size for sgd routine
    optim_params.Display     = 'off';     % opt routine output
    optim_params.monitor     = 'iter';    % save opt routine output

    % both matlab and mark schmidt options
    optim_params.max_iter    = 1000;        
    optim_params.maxIter     = 1000;        
    optim_params.maxFunEvals = 2000;      % mostly for sd in minFunc
    optim_params.optTol      = 1e-6;      % tol on first order optimality (max(abs(grad))
    optim_params.progTol     = 1e-16;     % tol on function/parameter values
    optim_params.TolX        = 1e-10;     % tol on progress in terms of function/parameter changes
    optim_params.TolFun      = 1e-7;      % tol on first order optimality

    % just mark schmidt options
    optim_params.Method = 'lbfgs';

    % just matlab options
    optim_params.Algorithm  = 'quasi-newton';
    optim_params.HessUpdate = 'steepdesc'; % bfgs default incredibly slow
    optim_params.GradObj    = 'on';
    optim_params.DerivativeCheck = 'off';
    optim_params.numDiff    = 0;

    end % method
    
end
   
end


