classdef RLVM
    
% Object-oriented implementation of an autoencoder with a single hidden 
% layer that also incorporates a basic stimulus processing model 
%
% Reference:
%   NIM
%   Andrew Ng
%
% Author: Matt Whiteway
%   11/05/16

% TODO
%   - throw error if num_stim_subs > 0 but no stim_params are provided
%   - interface with stim subunit in general sucks
%   - speed up calcualations when all subunits have same nonlinearity
%   - make what 'integration layer' option is clearer; input layer counts?
%   - fitting offsets in stim subunit should be default

properties
    integration_layer
    fit_params          % struct defining the fitting parameters
        % fit_auto
        % fit_stim_individual
        % fit_stim_shared
        % noise_dist
    optim_params
        % opt_routine
        % batch_size
        % max_iter
        % display
        % monitor
        % deriv_check
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
    layers              % array of layer object for autoencoder portion
        % weights
        % biases
        % ext_weights
        % act_func
        % reg_lambdas
        % init_params
        %   rng_state
        %   init_weights
        % latent vars
    fit_history         % struct array that saves result of each fitting step
        % fit_type
        % r2
        % first_order_opt
        % exit_msg
end

properties (Hidden)
    version = '2.0';        % version number
    date = date;            % date of fit
    min_pred_rate = 1e-5;   % min val for logs
    max_g = 30;             % max val for exponentials
    
    % user options
    allowed_noise_dists   = {'gauss', 'poiss', 'bern'};
    allowed_auto_regtypes = {'l2_weights', 'l2_biases', 'd2t_hid'};
    allowed_auto_NLtypes  = {'lin', 'relu', 'sigmoid', 'softplus'};
    allowed_stim_regtypes = {'l2', 'd2t', 'd2x', 'd2xt'};    
    allowed_stim_NLtypes  = {'lin', 'relu', 'softplus'};
    allowed_init_types    = {'gauss', 'trunc_gauss', 'uniform', 'orth'};
    allowed_pretraining   = {'none', 'pca', 'pca-varimax', 'layer'};
    allowed_optimizers    = {'fminunc', 'minFunc', 'con'};
end

% methods that are defined in separate files
methods
    net                         = fit_weights_latent_vars(net, fit_struct);
    [net, weights, latent_vars] = fit_weights(net, fit_struct);
    [net, weights, latent_vars] = fit_latent_vars(net, fit_struct);
    [net, weights, latent_vars] = fit_alt(net, fit_struct)
end


%% ********************* constructor **************************************
methods
      
    function net = RLVM(layer_sizes, num_stim_subs, varargin)
    % net = RLVM(layer_sizes, num_stim_subs, kargs) 
    %
    % constructor function for an RLVM object; sets properties, including 
    % calling Layer and StimSubunit constructors (no regularization is set)
    %
    % INPUTS:
    %   layer_sizes:    vector defining number of units in each layer,
    %                   including input, hidden and output layers
    %   num_stim_subs:  number of stimulus subunits, whether shared or
    %                   individual
    %
    %   optional key-value pairs: [defaults]
    %       'noise_dist', string
    %           ['gauss'] | 'poiss' | 'bern'
    %           specifies noise distribution for cost function
    %       'act_funcs', cell array of strings, one for each (non-input)
    %           layer of the rlvm
    %           ['lin'] | 'relu' | 'sigmoid' | 'softplus'
    %           If all layers share the same act_func, use a single string
    %       'auto_init', cell array of strings, one for each layer of the
    %           rlvm
    %           ['trunc_gauss'] | 'gauss' | 'uniform' | 'orth'
    %           If all layers share the same auto_init, use a single string
    %       'int_layer', scalar
    %           specifies which layer integrates autoencoder input with
    %           stimulus input (Default: final layer)
    %
    %       'stim_params', struct array
    %           each entry has the form [num_lags, num_xpix, num_ypix] to
    %           define the dimensions of the stimulus
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
    % OUTPUT:
    %   net: initialized RLVM object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return 
    end
    
    assert(length(layer_sizes) > 1, ...
        'Must specify at least two layers')
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')

    % define defaults    
    noise_distribution = 'gauss';
    auto_init_filt = repmat({'trunc_gauss'}, 1, length(layer_sizes)-1);
    act_funcs = repmat({'relu'}, 1, length(layer_sizes)-1);
    act_funcs{end} = 'lin'; % for 'gauss' noise dist default
    stim_params = [];
    int_layer = [];
    
    % stim_subunit - default is gaussian noise, all subunits excitatory and
    % use first target
    stim_init_weights = repmat({'gauss'}, 1, num_stim_subs);
    mod_signs = ones(1, num_stim_subs);                      
    NL_types = repmat({'lin'}, 1, num_stim_subs);  
    x_targets = ones(1,num_stim_subs);
   
    
    % -------------------- PARSE INPUTS -----------------------------------
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            
            % layers
            case 'noise_dist'
                assert(ismember(varargin{i+1}, net.allowed_noise_dists),...
                    'Invalid noise distribution')
                noise_distribution = varargin{i+1};
            case 'act_funcs'
                assert(all(ismember(varargin{i+1}, net.allowed_auto_NLtypes)), ...
                    'Invalid layer nonlinearities')
                act_funcs = varargin{i+1};
                % if act_funcs is specified as a single string, default to 
                % using this act_func for all layers
                if ~iscell(act_funcs) && ischar(act_funcs)
                    act_funcs = cellstr(act_funcs);
                end
                if length(act_funcs) == 1 && length(layer_sizes) > 2
                    act_funcs = repmat(act_funcs,1,length(layer_sizes)-1);
                elseif length(act_funcs) ~= length(layer_sizes)-1
                    error('Invalid number of act_funcs')
                end
            case 'auto_init'
                assert(all(ismember(varargin{i+1}, net.allowed_init_types)), ...
                    'Invalid weight init types')
                auto_init_filt = varargin{i+1};
                % if auto_init_filt is specified as a single string, 
                % default to using this auto_init_filt for all layers
                if ~iscell(auto_init_filt) && ischar(auto_init_filt)
                    auto_init_filt = cellstr(auto_init_filt);
                end
                if length(auto_init_filt) == 1 && length(layer_sizes) > 2
                    auto_init_filt = repmat(auto_init_filt,1,length(layer_sizes)-1);
                elseif length(auto_init_filt) ~= length(layer_sizes)-1
                    error('Invalid number of auto_inits')
                end
            case 'int_layer'
                assert(varargin{i+1} > 0 && ...
                    varargin{i+1} <= length(layer_sizes),...
                    'Invalid integration layer specified')
                int_layer = varargin{i+1};

            % stim_subunit
            case 'stim_params'
                assert(~isempty(varargin{i+1}), ...
                    'Invalid stim param struct')
                stim_params = varargin{i+1};
            case 'mod_signs'
                assert(all(ismember(varargin{i+1},[-1,1])),...
                    'Invalid mod_sign')
                mod_signs = varargin{i+1};
                % if mod_signs is specified as a single number, default to
                % using this mod_sign for all subunits
                if length(mod_signs) == 1 && num_stim_subs > 1
                    mod_signs = repmat(mod_signs,1,num_stim_subs); 
                elseif length(mod_signs) ~= num_stim_subs
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
                if length(NL_types) == 1 && num_stim_subs > 1
                    NL_types = repmat(NL_types,1,num_stim_subs);
                elseif num_stim_subs == 0
                    
                elseif length(NL_types) ~= num_stim_subs
                    error('Invalid number of NL_types')
                end
            case 'x_targets'
%                 assert(all(ismember(varargin{i+1}, 1:num_stims)),...
%                     'Invalid x_target')
                x_targets = varargin{i+1};
                % if x_targets is specified as a single number, default to
                % using this x_target for all subunits
                if length(x_targets) == 1 && num_stim_subs > 1
                    x_targets = repmat(x_targets, 1, num_stim_subs); 
                elseif length(x_targets) ~= num_stim_subs
                    error('Invalid number of x_targets')
                end
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % define fit_params defaults
    if ~isempty(stim_params)
        % simple check: if the number of stim subs is an integer multiple
        % of the number of inputs, then we're probably fitting individual
        % stimulus subunits for each cell
        if mod(stim_params.num_outputs, layer_sizes(end)) == 0 && ...
                layer_sizes(end) ~= 1
            fit_stim_individual = 1;
            fit_stim_shared = 0;
        else
            fit_stim_individual = 0;
            fit_stim_shared = 1;
        end
    else
        fit_stim_individual = 0;
        fit_stim_shared = 0;
    end
    if length(layer_sizes) > 1     
        fit_auto = 1;
    else
        fit_auto = 0; 
    end 
   
    % -------------------- SET PROPERTIES ---------------------------------
    
    net.integration_layer = int_layer;
    % fit_params
    net.fit_params.fit_auto = fit_auto;
    net.fit_params.fit_stim_individual = fit_stim_individual;
    net.fit_params.fit_stim_shared = fit_stim_shared;
    net.fit_params.noise_dist = noise_distribution;
    % optim_params defaults
    net.optim_params = RLVM.set_init_optim_params();
    net.optim_params.deriv_check = 0;
    % fit_history
    net.fit_history = struct([]);    

    % -------------------- INITIALIZE SUBUNITS ----------------------------
    
    % autoencoder subunit; initialized w/o regularization
    % initialize empty array of Layer objects
    layers_(length(layer_sizes)-1,1) = Layer();
    num_ext_inputs = zeros(length(layer_sizes)-1,1);
    if ~isempty(int_layer)
        num_ext_inputs(int_layer) = num_stim_subs;
    end
    for n = 1:length(layer_sizes)-1
        if n == 1 && layer_sizes(n) == 0 
            assert(num_stim_subs > 0, ...
                'Must have a stimulus model if not using autoencoder')
        end
        layers_(n) = Layer(layer_sizes(n), layer_sizes(n+1), ...
                            auto_init_filt{n}, ... 
                            'act_func', act_funcs{n}, ...
                            'num_ext_inputs', num_ext_inputs(n));
        % inherit values
        layers_(n).allowed_auto_regtypes = net.allowed_auto_regtypes;
        layers_(n).allowed_auto_NLtypes = net.allowed_auto_NLtypes;
        layers_(n).min_pred_rate = net.min_pred_rate;
        layers_(n).max_g = net.max_g;       
    end
    net.layers = layers_;
    if strcmp(net.layers(end).act_func, 'softplus')
        net.layers(end).biases = -1.5*(1+net.layers(end).biases);
    end
    
    % stimulus subunits; initialize w/o regularization, loop through
    if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
        % initialize empty array of StimSubunit objects
        stim_subunits_(num_stim_subs,1) = StimSubunit();
        for n = 1:num_stim_subs
            stim_subunits_(n) = StimSubunit( ...
                                    stim_params(x_targets(n)), ...
                                    stim_init_weights{n}, ...
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
    %       'noise_dist', string
    %           ['gauss'] | 'poiss' | 'bern'
    %           specifies noise distribution for cost function
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
                    'fit_auto option must be set to 0 or 1')
                net.fit_params.fit_auto = varargin{i+1};
            case 'fit_stim_individual'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    'fit_stim_individual option must be set to 0 or 1')
                assert(~(varargin{i+1} == 1 ...
                    && net.fit_params.fit_stim_shared == 1), ...
                    'RLVM:invalidoption', ...
                    'cannot fit both individual and shared subunits')
                net.fit_params.fit_stim_individual = varargin{i+1};
            case 'fit_stim_shared'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    'fit_stim_shared option must be set to 0 or 1')
                assert(~(varargin{i+1} == 1 ...
                    && net.fit_params.fit_stim_individual == 1), ...
                    'RLVM:invalidoption', ...
                    'cannot fit both individual and shared subunits')
                net.fit_params.fit_stim_shared = varargin{i+1};
            case 'noise_dist'
                assert(ismember(varargin{i+1}, net.allowed_noise_dists),...
                    'Invalid noise distribution')
                net.fit_params.noise_dist = varargin{i+1};
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
    %           specify the optimization routine used for learning the
    %           weights and biases; see allowed_optimizers for options
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
    %       'deriv_check', boolean
    %           specifies numerical derivative checking
    %
    % OUTPUTS:
    %   net: updated RLVM object
    
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
                    'invalid parameter for display')
                net.optim_params.Display = varargin{i+1};
            case 'max_iter'
                assert(varargin{i+1} > 0, ...
                    'max number of iterations must be greater than zero')
                net.optim_params.max_iter = varargin{i+1};
                net.optim_params.maxIter = varargin{i+1};
                net.optim_params.maxFunEvals = 2*varargin{i+1};
            case 'monitor'
                assert(ismember(varargin{i+1}, {'off', 'iter', 'batch', 'both'}), ...
                    'invalid parameter for monitor')
                net.optim_params.monitor = varargin{i+1};
            case 'deriv_check'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    'deriv_check option must be set to 0 or 1')
                net.optim_params.deriv_check = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    end % method
    
    
    function net = set_reg_params(net, reg_target, varargin)
    % net = net.set_reg_params(reg_target, kargs)
    % example: net = net.set_reg_params('layer', 'l2_weights', 10)
    %
    % Takes a sequence of key-value pairs to set regularization parameters
    % for either the stimulus model or the autoencoder model in an RLVM
    % object. Note that both the StimSubunit class and the Layer 
    % class have an equivalent method; the main usefulness of this method 
    % is to quickly update the reg_params structure for ALL stimulus 
    % subunits and layers
    %
    % INPUTS:
    %   reg_target:     'layer' | 'stim'
    %                   string specifying which model components to apply 
    %                   the specified reg params to
    %
    %   optional key-value pairs:
    %       'subs', vector
    %           specify set of subunits/layers to apply the new reg_params
    %           to (default: all subunits/layers)
    %       'layers', vector
    %           specify set of subunits/layers to apply the new reg_params
    %           to (default: all subunits/layers)
    %       'lambda_type', scalar
    %           'auto' lambda_types:
    %           'l2_weights' | 'l2_biases' | 'd2t_hid'
    %           
    %           'stim' lambda_types:
    %           'l2' | 'd2t' | 'd2x' | 'd2xt'
    %
    %           first input is a string specifying the type of 
    %           regularization, followed by a scalar giving the associated 
    %           regularization value, which will be applied to the layers
    %           or stimulus subunits specified by 'subs'. If different 
    %           lambda values are desired for different layers/subunits, 
    %           for now this method will have to be called for each update
    %   
    % OUTPUTS:
    %   net: updated RLVM object
    
    % check for appropriate number of inputs
    assert(ismember(reg_target, {'stim', 'layer'}), ...
        'Must specify which subunit to update')
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    % parse inputs
    switch reg_target
        case 'layer'
            % look for layers; if none specified, the reg_params will be 
            % applied to ALL layers
            layers_loc = find(strcmp(varargin, 'layers'));
            if ~isempty(layers_loc)
                assert(all(ismember(varargin{layers_loc+1}, ...
                                    1:length(net.layers))), ...
                    'invalid target layers specified')
                layers_inds = varargin{layers_loc+1};
                % remove sub_inds from varargin; will be passed to another 
                % method
                varargin(layers_loc) = [];
                varargin(layers_loc) = []; % equiv to layers_loc+1 after delete 
            else
                % default is to update all subunits
                layers_inds = 1:length(net.layers); 
            end               
            % update reg_params for all desired stim subunits
            for i = 1:length(layers_inds)
                net.layers(layers_inds(i)) = ...
                    net.layers(layers_inds(i)).set_reg_params(varargin{:}); 
            end
        case 'stim'
            % look for subs; if none specified, the reg_params will be 
            % applied to ALL subunits
            subs_loc = find(strcmp(varargin, 'subs'));
            if ~isempty(subs_loc)
                assert(all(ismember(varargin{subs_loc+1}, ...
                                    1:length(net.stim_subunits))), ...
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
        otherwise
            error('Invalid reg_target; must be auto or stim')
    end
    
    end % method
	
end
    
%% ********************* getting methods **********************************
methods

    function [a, z, gint, fgint] = get_mod_internals(net, varargin)
    % [a, z, gint, fgint] = net.get_mod_internals(varargin);
    %
    % Evaluates current RLVM object and returns activation values for
    % different layers of the model
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'pop_activity'  
    %           num_cells x T matrix of neural activity
    %       'Xstims', cell array
    %           cell array of stimuli, each of which is T x filt_len
    %       'inputs'  
    %           num_inputs x T matrix of external input activity
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   z           num_layers x 1 cell array, each cell containing a
    %               matrix of the signal before being passed through
    %               the activation function of the layer
    %   a           same as z, except value of signal after being
    %               passed through the activation function
    %   gint        T x num_subunits matrix of subunit filter outputs
    %   fgint       same as gint, except value of filtered output after
    %               being passed through the subunit's nonlinearity

    % define defaults
    pop_activity = [];
    Xstims = [];
    inputs = [];
    indx_tr = NaN; % NaN means we use all available data
    T = 0;
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'pop_activity'
                % error checking later
                pop_activity = varargin{i+1};
                if ~isempty(pop_activity)
                    T = size(pop_activity, 2);
                end
            case 'xstims'
                % error checking later
                Xstims = varargin{i+1};
                if ~isempty(Xstims)
                    T = size(Xstims{1}, 1);
                end
            case 'inputs'
                % error checking later
                inputs = varargin{i+1};
                if ~isempty(inputs)
                    T = size(inputs, 2);
                end
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:T)) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
   
    % make sure the proper data is present
    if ~isempty(net.layers(1).weights) && (isempty(pop_activity) && isempty(inputs))
        error('must specify pop_activity to fit model')
    end
    if ~isempty(net.stim_subunits) && isempty(Xstims)
        error('must specify Xstims to fit model')
    end
    % TODO error-checkin on input
    
    % use indx_tr
    if ~isnan(indx_tr)
        if ~isempty(pop_activity)
            pop_activity = pop_activity(:,indx_tr);
        end
        if ~isempty(Xstims)
            for i = 1:length(Xstims)
                Xstims{i} = Xstims{i}(indx_tr,:);
            end
        end
        if ~isempty(inputs)
            inputs = inputs(:, indx_tr);
        end
        T = length(indx_tr);
    else
        if ~isempty(pop_activity)
            T = size(pop_activity, 2);
        elseif ~isempty(Xstims)
            T = size(Xstims{1}, 1);
        end
    end
    
    % do we fit input model or auto model?
    if ~isempty(inputs)
        fit_auto = 0;
    else
        fit_auto = 1;
    end
        
    % get internal generating signals - stim
    if ~isempty(net.stim_subunits)
        num_subunits = length(net.stim_subunits);
        if net.fit_params.fit_stim_shared
            gint = zeros(T, num_subunits);
            fgint = zeros(T, num_subunits);
            for i = 1:num_subunits
                [fgint(:,i), gint(:,i)] = ...
                    net.stim_subunits(i).get_model_internals(Xstims);
            end
        elseif net.fit_params.fit_stim_individual
            gint = cell(num_subunits, 1);
            fgint = cell(num_subunits, 1);
            for i = 1:num_subunits
                [fgint{i}, gint{i}] = ...
                    net.stim_subunits(i).get_model_internals(Xstims);
            end
        end
    else
        gint = [];
        fgint = [];
    end
    
    % get internal generating signals - auto
    z = cell(length(net.layers),1);
    a = cell(length(net.layers),1);
    for i = 1:length(net.layers)
        if i == 1
            if ~isempty(net.layers(1).weights)
                if fit_auto
                    % auto model
                    z{i} = bsxfun(@plus, net.layers(i).weights*pop_activity, ...
                                     net.layers(i).biases);
                else
                    % nn model
                    z{i} = bsxfun(@plus, net.layers(i).weights*inputs, ...
                                     net.layers(i).biases);
                end
            else
                % just stimulus model
                z{i} = repmat(net.layers(i).biases, 1, T);
            end
        else
            z{i} = bsxfun(@plus, net.layers(i).weights*a{i-1}, ...
                                 net.layers(i).biases);
        end
        if i == net.integration_layer
            z{i} = z{i} + net.layers(i).ext_weights * fgint';
        end
        if i == length(net.layers) && net.fit_params.fit_stim_individual
            for j = 1:num_subunits
                z{i} = z{i} + net.stim_subunits(j).mod_sign * fgint{j}';
            end
        end
        a{i} = net.layers(i).apply_act_func(z{i});
    end
    
    end % method
    
    
    function [r2, LL_struct] = get_r2(net, pop_activity, pred_activity)
    % r2 = net.get_r2(pop_activity, pred_activity);
    %
    % Evaluates current RLVM object using (pseudo) R2
    %
    % INPUTS:
    %   pop_activity    num_cells x T matrix of neural activity
    %   pred_activity   num_cells x T matrix of predicted neural activity
    %
    %   optional key-value pairs:
    %       'Xstims', cell array
    %           cell array of stimuli, each of which is T x filt_len
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   r2              num_cells x 1 vector of r2s
    %   LL_struct       contains LL, LLnull and LLsat
    
    T = size(pop_activity,2);
    mean_activity = mean(pop_activity,2) * ones(1,T);
    
    switch net.fit_params.noise_dist
        case 'gauss'
            LL = sum((pred_activity - pop_activity).^2, 2);
            LLnull = sum((pop_activity - mean_activity).^2, 2);
            LLsat = zeros(size(pop_activity,1),1);
        case 'poiss'
            LL = -sum(pop_activity.*log(pred_activity) - pred_activity, 2);
            LLnull = -sum(pop_activity.*log(mean_activity) - mean_activity, 2);
            LLnull(sum(mean_activity,2)==0) = NaN;
            LLsat = pop_activity.*log(pop_activity);
            LLsat(pop_activity==0) = 0;
            LLsat = -sum(LLsat - pop_activity, 2);
        case 'bern'
            LL1 = pop_activity.*log(pred_activity);
            LL1(pred_activity==0) = 0;
            LL2 = (1-pop_activity).*log(1-pred_activity);
            LL2(pred_activity==1) = 0;
            LL = -sum(LL1 + LL2, 2);
            LLnull = -sum(pop_activity.*log(mean_activity) + ...
                      (1-pop_activity).*log(1-mean_activity), 2);
            LLnull(sum(mean_activity,2)==0) = NaN;
            LLsat = zeros(size(pop_activity,1),1);
        otherwise
            error('Invalid noise distribution')
    end
    
    r2 = 1 - (LLsat-LL)./(LLsat-LLnull);

    if nargout > 1
        LL_struct.LL = LL;
        LL_struct.LLnull = LLnull;
        LL_struct.LLsat = LLsat;
    end
    
    end % method
    
    
    function [r2, LL_struct] = get_sloo_r2(net, pop_activity, varargin)
    % r2 = net.get_sloo_r2(pop_activity, kargs)
    %
    % Evaluates r2 using a simple leave-one-out procedure; for each cell,
    % the first layer weight of the autoencoder portion of the network is
    % set to zero, and the resulting predicted activity for that cell is
    % compared to the true activity.
    %
    % INPUTS:
    %   'pop_activity'  num_cells x T matrix of neural activity
    %
    %   optional key-value pairs:
    %       'Xstims', cell array
    %           cell array of stimuli, each of which is T x filt_len
    %       'inputs', matrix
    %           num_inputs x T matrix of non-stim inputs
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   r2:             num_cells x 1 vector of rsquared values 
    %   LL_struct       contains LL, LLnull and LLsat
    
    assert(~isempty(net.layers(1).weights), ...
        'Cannot calculate r2s using sloo method without an autoencoder')
    assert(size(net.layers(1).weights, 2) == size(net.layers(end).weights, 1), ...
        'Cannot calculate r2s using sloo method without an autoencoder')

    % define defaults
    Xstims = [];
    inputs = [];
    indx_tr = NaN; % NaN means we use all available data
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'xstims'
                if ~isempty(Xstims)
                    assert(iscell(varargin{i+1}), ...
                        'Xstims must be a cell array')
                end
                Xstims = varargin{i+1};
            case 'inputs'
                if ~isempty(inputs)
                    assert(size(varargin{i+1}, 2) == size(pop_activity, 2), ...
                        'Input matrix size inconsistent with population activity')
                end
                inputs = varargin{i+1};
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(pop_activity, 2))) ...
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
        pop_activity = pop_activity(:,indx_tr);
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:);
        end
        if ~isempty(inputs)
            inputs = inputs(:, indx_tr);
        end
    end
    
    num_cells = size(net.layers(1).weights, 2);
    r2 = zeros(num_cells, 1);
    LL_struct.LL = zeros(num_cells, 1);
    LL_struct.LLnull = zeros(num_cells, 1);
    LL_struct.LLsat = zeros(num_cells, 1);
    
    for c = 1:num_cells
        
        temp_net = net;
        temp_net.layers(1).weights(:,c) = 0;
    
        % get activation values for all model components
        a = temp_net.get_mod_internals( ...
                        'pop_activity', pop_activity, ...
                        'inputs', inputs, ...
                        'Xstims', Xstims);

        % evaluate pseudo-r^2
        [temp_r2, temp_LL_struct] = temp_net.get_r2(pop_activity, a{end});
        r2(c) = temp_r2(c);
        LL_struct.LL(c) = temp_LL_struct.LL(c);
        LL_struct.LLnull(c) = temp_LL_struct.LLnull(c);
        LL_struct.LLsat(c) = temp_LL_struct.LLsat(c);
        
    end
    
    end % method
    
    
    function [r2, cost_func, mod_internals, mod_reg_pen, LL_struct] = ...
                get_model_eval(net, pop_activity, varargin)
    % [r2, cost_func, mod_internals, mod_reg_pen] = ...
    %                           net.get_model_eval(pop_activity, kargs)
    %
    % Evaluates current RLVM object and returns relevant model data
    % like goodness-of-fit (r2 or pseudo-r2), value of the cost function,
    % regularization information, etc.
    %
    % INPUTS:
    %   'pop_activity'  num_cells x T matrix of neural activity
    %
    %   optional key-value pairs:
    %       'Xstims', cell array
    %           cell array of stimuli, each of which is T x filt_len
    %       'inputs', matrix
    %           num_inputs x T matrix of non-stim inputs
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %       'r2_type', string
    %           ['full'] | 'sloo'
    %           'full' computes r2 by filtering data through the network
    %           and comparing predicted values with true values. 'sloo', or
    %           'simple leave-one-out', computes r2 for each cell
    %           individually by zeroing out its input value and comparing
    %           the resulting predicted values with true values.
    %
    % OUTPUTS:
    %   r2:             num_cells x 1 vector of rsquared values 
    %   cost_fun:       scalar value of unregularized cost function
    %   mod_internals:  struct with fields
    %       z           num_layers x 1 cell array, each cell containing a
    %                   matrix of the signal before being passed through
    %                   the activation function of the layer
    %       a           same as z, except value of signal after being
    %                   passed through the activation function
    %       gint        T x num_subunits matrix of subunit filter outputs
    %       fgint       same as gint, except value of filtered output after
    %                   being passed through the subunit's nonlinearity
    %   mod_reg_pen:        
    %       layer_reg_pen struct containing penalties due to different regs
    %       stim_reg_pen  struct containing penalties due to different regs
    %   LL_struct       contains LL, LLnull and LLsat

    % define defaults
    Xstims = [];
    inputs = [];
    indx_tr = NaN; % NaN means we use all available data
    r2_type = 'full';
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'xstims'
                if ~isempty(Xstims)
                    assert(iscell(varargin{i+1}), ...
                        'Xstims must be a cell array')
                end
                Xstims = varargin{i+1};
            case 'inputs'
                if ~isempty(inputs)
                    assert(size(varargin{i+1}, 2) == size(pop_activity, 2), ...
                        'Input matrix size inconsistent with population activity')
                end
                inputs = varargin{i+1};
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(pop_activity, 2))) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            case 'r2_type'
                assert(ismember(varargin{i+1}, {'full', 'sloo'}), ...
                    'Invalid r2_type');
                r2_type = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
   
    % use indx_tr
    if ~isnan(indx_tr)
        pop_activity = pop_activity(:,indx_tr);
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:);
        end
        if ~isempty(inputs)
            inputs = inputs(:, indx_tr);
        end
    end
    
    % get activation values for all model components
    [a, z, gint, fgint] = net.get_mod_internals( ...
        'pop_activity', pop_activity, ...
        'Xstims', Xstims, ...
        'inputs', inputs);
        
    % evaluate pseudo-r^2
    switch r2_type
        case 'full'
            [r2, LL_struct] = net.get_r2(pop_activity, a{end});
        case 'sloo'
            [r2, LL_struct] = net.get_sloo_r2(pop_activity, ...
                                              'inputs', inputs, ...
                                              'Xstims', Xstims);
    end
    
    % evaluate cost_func
    switch net.fit_params.noise_dist
        case 'gauss'
            Z = 2 * numel(pop_activity);
            LL = sum((a{end} - pop_activity).^2, 2);
        case 'poiss'
            Z = sum(pop_activity(:));
            LL = -sum(pop_activity.*log(a{end}) - a{end}, 2);
        case 'bern'
            Z = numel(pop_activity);
            LL1 = pop_activity.*log(a{end});
            LL1(a{end}==0) = 0;
            LL2 = (1-pop_activity).*log(1-a{end});
            LL2(a{end}==1) = 0;
            LL = -sum(LL1 + LL2, 2);
        otherwise
            error('Invalid noise distribution')
    end
    cost_func = sum(LL) / Z;
    
    % get regularization penalites - layer
    layer_reg_pen = [];
    for i = 1:length(net.layers)
        layer_reg_pen = ...
            cat(1, layer_reg_pen, net.layers(i).get_reg_pen(pop_activity));
    end
    % get regularization penalties - stim
    stim_reg_pen = [];
    for i = 1:length(net.stim_subunits)
        stim_reg_pen = ...
            cat(1, stim_reg_pen, net.stim_subunits(i).get_reg_pen());
    end
    
    % put internal generating functions in a struct if desired in output
    if nargout > 2
        mod_internals.z = z;
        mod_internals.a = a;
        mod_internals.gint = gint;
        mod_internals.fgint = fgint;
    end
    % put reg penalties in a struct if desired in output
    if nargout > 3
        mod_reg_pen = struct('layer_reg_pen', layer_reg_pen, ...
                             'stim_reg_pen', stim_reg_pen);
    end

    end % method
    
end

%% ********************* fitting methods **********************************
methods
    
    function [net, varargout] = fit_model(net, fit_type, varargin)
    % net = net.fit_model(fit_type, kargs)
    %
    % Checks inputs and farms out parameter fitting to other methods 
    % depending on what type of model fit is desired
    %
    % INPUTS:
    %   fit_type:   'weights' | 'inputs' | 
    %               'mml_weights' | 'mml_latent_vars' | 'mml_alt'
    %
    %   optional key-value pairs:
    %       'pop_activity', matrix
    %           num_cells x T matrix of neural activity; required if weight
    %           matrix in first layer of model is not empty
    %       'Xstims', cell array
    %           stimulus covariate matrices; required if stim_subunits is
    %           not empty
    %       'inputs', matrix
    %           num_inputs x T matrix of inputs; required if number of 
    %           input nodes is different from number of output nodes
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           fitting
    %       'pretraining', string
    %           ['none'] | 'pca' | 'pca-varimax' | 'layerwise'
    %           specifies how a model with a full autoencoder component is
    %           initialized
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
    %           ['auto'] | 'init'
    %           used by fit_alt to initialize model; 'auto' fits the
    %           autoencoder, 'init' uses values stored in 'init_weights' 
    %           and 'init_latent_vars'
    %       'first_fit', string
    %           ['latent_vars'] | 'weights'
    %           used by fit_alt to determine which set of parameters to fit
    %           first
    %
    % OUTPUTS:
    %   net: updated RLVM object
    
    assert(ismember(fit_type, {'weights', 'mml_weights', ...
                               'mml_latent_vars', 'mml_alt', 'inputs'}), ...
        'Invalid fit_type specified')
    
    % DEFINE DEFAULTS
    % put all relevant data into fitting_struct, which acts as a common
    % data structure to all fitting routines
    pop_activity = [];
    Xstims = [];
    inputs = [];
    indx_tr = NaN;              % train on all data
    pretraining = 'none';       % no pretraining of weights
    init_weights = 'model';     % init mml weights from model
    init_latent_vars = 'model'; % init mml lvs from model
    init_method = 'auto';       % init model with autoencoder
    first_fit = 'latent_vars';  % fit lvs before weights
    
    % PARSE OPTIONAL INPUTS
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'xstims'
                % error checking later
                Xstims = varargin{i+1};
            case 'pop_activity'
                % error checking later
                pop_activity = varargin{i+1};
            case 'inputs'
                % error checking later
                inputs = varargin{i+1};
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(pop_activity,2))) ...
                    && ~any(isnan(varargin{i+1})), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            case 'pretraining'
                assert(ismember(varargin{i+1}, net.allowed_pretraining), ...
                    'invalid pretraining option')
                pretraining = varargin{i+1};
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
            case 'init_method'
                assert(ismember(varargin{i+1}, {'auto', 'init'}), ...
                    'Improper init_method specified')
                init_method = varargin{i+1};
            case 'first_fit'
                assert(ismember(varargin{i+1}, {'weights', 'latent_vars'}), ...
                    'Improper fit_first specified')
                first_fit = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % make sure the proper data is present
    if ~isempty(net.layers(1).weights) && isempty(pop_activity)
        error('must specify pop_activity to fit model')
    end
    if ~isempty(net.stim_subunits) && isempty(Xstims)
        error('must specify Xstims to fit model')
    end
    if strcmp(fit_type, 'inputs') && isempty(inputs)
        error('must specify an input matrix to fit model')
    end
    
    % create fitting struct
    if isnan(indx_tr)
        fit_struct.pop_activity = pop_activity;
        fit_struct.Xstims = Xstims;
        fit_struct.inputs = inputs;
    else
        if ~isempty(pop_activity)
            fit_struct.pop_activity = pop_activity(:,indx_tr);
        else
            fit_struct.pop_activity = [];
        end
        if ~isempty(Xstims)
            for i = 1:length(Xstims)
                fit_struct.Xstims{i} = Xstims{i}(indx_tr,:);
            end
        else
            fit_struct.Xstims = [];
        end
        if ~isempty(inputs)
            fit_struct.inputs = inputs(:,indx_tr);
        else
            fit_struct.inputs = [];
        end
    end
    clear pop_activity
    clear Xstims
    clear inputs
    fit_struct.init_weights = init_weights;
    fit_struct.init_latent_vars = init_latent_vars;
    fit_struct.init_method = init_method;
    fit_struct.first_fit = first_fit;
    
    % check consistency between inputs
    net.check_fit_struct(fit_struct, fit_type); 
    
    % pretrain model
    if ~strcmp(pretraining, 'none')
        assert(~isempty(net.layers(1).weights), ...
            'Cannot pretrain a stimulus-only model')
        if strcmp(fit_type, 'inputs')
            net = net.pretrain(fit_struct.inputs, pretraining);
        else
            net = net.pretrain(fit_struct.pop_activity, pretraining);
        end
    end
    
    % fit model
    if any(strcmp(fit_type, {'weights', 'inputs'}))
        net = net.fit_weights_latent_vars(fit_struct);
%     elseif strcmp(fit_type, 'inputs')
%         net = net.fit_weights_latent_vars_nn(fit_struct);
    elseif strcmp(fit_type, 'mml_weights')
        [net, weights, latent_vars] = net.fit_weights(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    elseif strcmp(fit_type, 'mml_latent_vars')
        [net, weights, latent_vars] = net.fit_latent_vars(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    elseif strcmp(fit_type, 'mml_alt')
        [net, weights, latent_vars] = net.fit_alt(fit_struct);
        varargout{1} = weights;
        varargout{2} = latent_vars;
    else
        error('Invalid fit_type')
    end
    
    end % method
    
    
    function net = pretrain(net, data, pretraining)
    % net = net.pretrain(data, pretraining)
	% 
    % Performs a layer-wise pretraining of the model in order to speed up
    % convergence during model fitting
    % For a symmetric network, just train encoding weights and set decoding
    % weights to be transposes
    % For a non-symmetric network, train the first half of the layers, set
    % the decoding weights in the second half to be transposes, and leave
    % the middle layer to be random weights
    %
	% INPUTS:
    %   data:           num_cells x T matrix of neural activity
    %   pretraining:    string specifying type of desired pretraining
    %
	% OUTPUTS:
    %   net:            updated RLVM object
    
    % get number of units per layer
    num_layers = length(net.layers);
    num_nodes = zeros(num_layers+1,1);
    for i = 1:num_layers
        num_nodes(i) = size(net.layers(i).weights,2);
    end
    num_nodes(num_layers+1) = size(net.layers(num_layers).weights,1);
    
    % determine properties of network
    if all(num_nodes == flipud(num_nodes))
        net_symmetric = 1;
    else
        net_symmetric = 0;
    end
    if num_layers > 1
        if mod(num_layers-1, 2) == 0
            middle_layer = (num_layers-1)/2 + 1;
        else
            middle_layer = ceil((num_layers-1)/2);
        end
%         [~, middle_layer] = min(num_nodes(2:end)); % don't include data layer
    else
        middle_layer = 0;
    end
    
    % set initial params
    temp_data = data;
    
    for layer = 1:middle_layer
        
        % update params
        num_hid = num_nodes(layer+1);
        
        switch pretraining
            case 'pca'                
                temp_data = bsxfun(@minus, temp_data, mean(temp_data,2));
                [temp_weights,~,~] = svd(temp_data,'econ');
                temp_weights = temp_weights(:,1:num_hid)';
                temp_data = temp_weights*temp_data;
            case 'pca-varimax'
                temp_data = bsxfun(@minus, temp_data, mean(temp_data,2));
                [temp_weights,~,~] = svd(temp_data,'econ');
                % only rotate factors if subspace is greater than 1-d
                if num_hid > 1
                    try
                        temp_weights = rotatefactors(temp_weights(:,1:num_hid), ...
                                        'Method', 'varimax')';
                    catch ME
                        % identifier: 'stats:rotatefactors:IterationLimit'
                        try
                            temp_weights = rotatefactors(temp_weights(:,1:num_hid), ...
                                        'Method', 'varimax', ...
                                        'reltol', 1e-3, ...
                                        'maxit', 2000)';
                        catch ME
                            temp_weights = temp_weights(:,1:num_hid)';
                        end
                    end
                else
                    temp_weights = temp_weights(:,1)';
                end
                temp_weights = bsxfun(@times, temp_weights, sign(mean(temp_weights,2)));
                temp_data = temp_weights*temp_data;
            case 'layerwise'
                
                error('TODO')
                
%                 if ~strcmp(net.optim_params.Display, 'off')
%                     fprintf('Pretraining layer %g of %g\n', layer, middle_layer)
%                 end
                
                % initialize model
                init_params = RLVM.set_init_params(temp_data);
                net0 = RLVM(init_params, ...
                            'num_hid_nodes', num_hid, ...
                            'init_weights', 'gauss');

                % specify additional net_params
                net0 = net0.set_net_params( ...
                            'act_func_hid', 'relu', ...
                            'act_func_out', 'lin');

                net0 = net0.set_reg_params( ...
                            'l2_weights', 1e-5, ...
                            'l2_biases', 1e-5);

                net0 = net0.set_optim_params( ...
                            'display', 'off', ...
                            'deriv_check', 0, ...
                            'maxIter', 5000);

                % fit
                net0 = net0.fit_weights(temp_data);
                
                temp_weights = net0.weights{1};
                [~,a] = net0.get_model_internals(temp_data);
                temp_data = a{1};
                
            otherwise
                error('Invalid pretraining string specified')
        end   
        
        net.layers(layer).weights = temp_weights;
        if net_symmetric
            net.layers(end-layer+1).weights = temp_weights';
        end
        
    end
    end % method
    
end

%% ********************* display methods **********************************
methods
    
    function fig_handle = disp_stim_filts(net, varargin)
    % fig_handle = net.disp_stim_filts(<cell_num>)
    %
    % Plots stimulus filters of various subunits for a given cell
    %
    % INPUTS:
    %	cell_num:			optional; cell number for plotting
    %
    % OUTPUTS:
    %   fig_handle:         handle of created figure

    % check inputs
    if ~isempty(varargin)
        assert(ismember(varargin{1}, 1:net.num_cells), ...
            'Invalid cell index')
        cell_num = varargin{1};
    else
        cell_num = [];
    end
    
    num_subs = length(net.stim_subunits);
    max_cols = 5;
    rows = ceil(num_subs/max_cols);
    cols = min(max_cols, num_subs);

    % set figure position
    fig_handle = figure;
    set(fig_handle,'Units','normalized');
    set(fig_handle,'OuterPosition',[0.05 0.55 0.6 0.4]); %[left bottom width height]

    if net.stim_subunits(1).stim_params.num_outputs == net.num_cells
        % individual stimulus
        assert(~isempty(cell_num), ...
            'Must provide cell number for individual subunit models')
        for i = 1:num_subs
            subplot(rows, cols, i)
            net.stim_subunits(i).disp_filt(i, cell_num);
        end
    elseif net.stim_subunits(1).stim_params.num_outputs == 1
        % shared stimulus
        if isempty(varargin)
            % plot filters for each subunit and stim_weights
            counter = 0;
            for i = 1:num_subs
                counter = counter + 1;
                if mod(counter, cols+1) == 0
                    counter = counter + 1;
                end
                subplot(rows, cols+1, counter)
                net.stim_subunits(i).disp_filt(i);
            end
            extra_slots = (cols+1):(cols+1):rows*(cols+1);
            subplot(rows, cols+1, extra_slots)
            imagesc(net.stim_weights', ...
                [-max(abs(net.stim_weights(:))), max(abs(net.stim_weights(:)))]);
            title('Stim Weights', 'FontSize', 12)
            xlabel('Subunit Number')
            ylabel('Cell Number')
            set(gca,'YDir', 'normal', 'YAxisLocation', 'right')
            colormap(jet);
        else
            % plot filters for each subunit, weighted by stim_weights
            if strcmp([net.stim_subunits.NL_type], repmat('lin', 1, num_subs))
                % if subunits are all linear, plot their weighted combo
                weighted_combo = zeros(size(net.stim_subunits(1).filt));
                counter = 0;
                for i = 1:num_subs
                    counter = counter + 1;
                    if mod(counter, cols+1) == 0
                        counter = counter + 1;
                    end
                    subplot(rows, cols+1, counter)
                    net.stim_subunits(i).disp_filt(i, [], net.stim_weights(i, cell_num));
                    weighted_combo = weighted_combo ...
                                   + net.stim_weights(i, cell_num) ...
                                   * net.stim_subunits(i).filt;
                end
                % right now this only works for 2d filters of same size
                extra_slots = (cols+1):(cols+1):rows*(cols+1);
                subplot(rows, cols+1, extra_slots)
                imagesc(reshape(weighted_combo, net.stim_subunits(1).stim_params.dims), ...
                    [-max(abs(weighted_combo(:))), max(abs(weighted_combo(:)))]);
                title('Weighted Combo', 'FontSize', 12)
                set(gca,'YDir', 'normal')
                colormap(jet);
                % go back through and put all filters on same scale
                counter = 0;
                for i = 1:num_subs
                    counter = counter + 1;
                    if mod(counter, cols+1) == 0
                        counter = counter + 1;
                    end
                    subplot(rows, cols+1, counter)
                    set(gca, 'clim', [-max(abs(weighted_combo(:))), ...
                                       max(abs(weighted_combo(:)))]);
                end
            else
                % just plot weighted subunits
                for i = 1:num_subs
                    subplot(rows, cols, i)
                    net.stim_subunits(i).disp_filt(i, [], net.stim_weights(i, cell_num));
                end
            end
        end
    end
    
    end % method
     
    
    function fig_handle = disp_lvs(net, time_vals, layer, varargin)
    % fig_handle = net.disp_lvs(varargin);
    %
    % Plots lvs of current RLVM object
    %
    % INPUTS:
    %   time_vals:          1 x T vector of time points
    %   layer:              specify which layers lvs come from
    %
    %   optional key-value pairs:
    %       'pop_activity'
    %           num_cells x T matrix of neural activity
    %       'Xstims', cell array
    %           cell array of stimuli, each of which is T x filt_len
    %       'inputs'
    %           num_inputs x T matrix of external input activity
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   fig_handle:         handle of created figure
    
    % get lvs
    [a, ~, ~, ~] = get_mod_internals(net, varargin{:});
    lvs = a{layer};
    num_lvs = size(lvs, 1);
    
    if isempty(time_vals)
        time_vals = 1:size(a{1},2);
    end
    
    fig_handle = figure;
    for i = 1:num_lvs
        ax(i) = subplot(num_lvs, 1, i);
        plot(time_vals, lvs(i,:));
        title(sprintf('Layer %g, latent variable %g', layer, i))
        set(fig_handle.CurrentAxes, 'FontSize', 14)
        if i == num_lvs
            xlabel('Time')
        end
    end
    linkaxes(ax, 'x')
    
    end
        
    
end

%% ********************* hidden methods ***********************************
methods (Hidden)
  
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
        
        if ~isempty(fit_struct.pop_activity)
            % ensure first dimension is same across pop_activity and Xstims
            assert(size(fit_struct.pop_activity, 2) == size(fit_struct.Xstims{1}, 1), ...
                'time dim must be consisent across pop activity and Xstims')
        end
    end
    
    if net.fit_params.fit_auto
    end
    if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
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


