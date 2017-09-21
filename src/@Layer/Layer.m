classdef Layer
    
% Class implementing an individual layer of an RLVM object
%
% Reference:
%
% Author: Matt Whiteway
%   09/01/17

properties
    weights         % matrix of weights
    biases          % vector of biases
    ext_weights     % weights from external inputs (e.g. stim models)
    act_func        % string specifying layer activation function
    reg_lambdas     % struct defining regularization hyperparameters
    init_params     % struct saving initialization parameters
      % rng_state
      % init_weights
end

properties (Hidden)
	allowed_reg_types = {'l2_biases', 'l2_weights'};
    allowed_act_funcs = {'lin', 'relu', 'sigmoid', 'softplus'};
    allowed_init_types = {'gauss', 'trunc_gauss', 'uniform', 'orth'}
    min_pred_rate   % min val for logs
    max_g = 50      % max val for exponentials
end

%% ********************  constructor **************************************
methods
      
    function layer = Layer(num_in, num_out, init_method, varargin)
    % layer = Layer(num_in, num_out, init_method, kwargs);
    % 
    % Constructor for Layer class
    %
    % INPUTS:
    %   num_in (scalar): number of input nodes
    %   num_out (scalar): number of output nodes
    %   init_method (string): specifies a random initialization; see
    %       allowed_init_types for supported options; or a 2x1 cell array 
    %       containt weights and biases of appropriate dimensions
    %
    %   optional key-value pairs
    %       'act_func', string
    %           specifies activation function for layer; see 
    %           allowed_act_funcs for supported options (default: relu)
    %       'num_ext_inputs', scalar
    %           specifies number of external inputs to be integrated into
    %           this layer
    %
    % OUTPUTS:
    %   layer (Layer object): initialized Layer object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return
    end

    % parse inputs
    assert(num_out > 0, 'Must specify positive number of outputs')
    
    % define defaults
    act_func_ = 'relu';
    num_ext_inputs = 0;
   
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'act_func'
                assert(ismember(varargin{i+1}, layer.allowed_act_funcs),...
                    'Invalid activation function "%s"', varargin{i+1})
                act_func_ = varargin{i+1};
            case 'num_ext_inputs'
                assert(~isempty(varargin{i+1}), ...
                    'Invalid number of external inputs specified')
                num_ext_inputs = varargin{i+1};
            otherwise
                error('Invalid input flag "%s"', varargin{i});
        end
        i = i + 2;
    end
    
    % initialize weights
    [weights_, biases_, init_params_] = Layer.set_init_weights_stat(...
        init_method, num_in, num_out);
        
    % set properties
    layer.weights = weights_;
    layer.biases = biases_;
    layer.act_func = act_func_;
    layer.reg_lambdas = Layer.init_reg_lambdas(); % init all to 0s
    layer.init_params = init_params_;
    if num_ext_inputs == 0    
        layer.ext_weights = [];
    else
        layer.ext_weights = 0.1 * randn(num_out, num_ext_inputs);
    end
    
    end

end
%% ********************  setting methods **********************************
methods
    
    function layer = set_layer_params(layer, varargin)
    % layer = layer.set_layer_params(kwargs)
    %
    % Takes a sequence of key-value pairs to set parameters for a Layer
    % object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'act_func', string
    %           'lin' | 'relu' | 'sigmoid' | 'softplus'
    %           specifies hidden layer activation functions
    %
    % OUTPUTS:
    %   layer (Layer object): updated Layer object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'act_func'
                assert(ismember(varargin{i+1}, layer.allowed_act_funcs), ...
                    'Invalid activation function "%s"', varargin{i+1});
                layer.act_func = varargin{i+1};
            otherwise
                error('Invalid input flag "%s"', varargin{i})
        end
        i = i + 2;
    end
    
    end % method
    
    
    function layer = set_reg_params(layer, varargin)
    % layer = layer.set_reg_params(kwargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters 
    % for a Layer object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'reg_type', scalar
    %           'l2_weights' | 'l2_biases'
    %
    % OUTPUTS:
    %   layer (Layer object): updated Layer object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
			case 'l2_weights'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                layer.reg_lambdas.l2_weights = varargin{i+1};
            case 'l2_biases'
                assert(varargin{i+1} >= 0, ...
                    'reg value must be nonnegative')
                layer.reg_lambdas.l2_biases = varargin{i+1};
            otherwise
                error('Invalid input flag "%s"', varargin{i});
        end
        i = i + 2;
    end
    
    end % method
    
    
    function layer = set_init_weights(layer, init_weights)
    % layer = layer.set_init_weights(init_weights)
    %
    % Sets weights and biases properties of Layer object
    %
    % INPUTS:
    %   init_weights (string): specifies a random initialization; see
    %       allowed_init_types for supported options; or a 2x1 cell array
    %       containing weights and biases of appropriate dimensions
    %
    % OUTPUT:
    %   layer (Layer object): updated Layer object
    
    % call static set_init_weights used in constructor
    [weights_, biases_, init_params_] = Layer.set_init_weights_stat(...
        init_weights, ...
        layer.num_in, ...
        layer.num_out);

    % set properties
    layer.weights = weights_;
    layer.biases = biases_;
    layer.init_params = init_params_;
    
    end % method
    
end
%% ********************  getting methods **********************************
methods
    
    function reg_pen = get_reg_pen(layer, sig, varargin)
    % reg_pen = layer.get_reg_pen(pop_activity, kwargs)
    %
    % Gets regularization penalties on layer weights and biases
    %
    % INPUTS:
    %   sig (num_in x T matrix)
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default: use all data)
    %
    % OUTPUTS:
    %   reg_pen (struct): contains penalties due to different regs
    
    % define defaults
    indx_tr = NaN; % NaN means we use all available data
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1},1:size(sig,2))) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('Invalid input flag "%s"', varargin{i});
        end
        i = i + 2;
    end
    
    % use indx_tr
    if ~isnan(indx_tr)
        sig = sig(:, indx_tr);
    end
    
    % set aside constants to keep things clean
    lambda_w = layer.reg_lambdas.l2_weights;
    lambda_b = layer.reg_lambdas.l2_biases;
    
    % get penalty terms
    % l2 on weights
    reg_pen.l2_weights = 0.5*lambda_w*sum(sum(layer.weights.^2)); 
    % l2 on biases
    reg_pen.l2_biases = 0.5*lambda_b*sum(layer.biases.^2);
    
    end % method
    
end
%% ********************  hidden methods ***********************************
methods (Hidden)
    
    function sig = apply_act_func(layer, sig)
    % sig = layer.apply_act_func(sig)
    %
    % internal function that applies activation function of the nodes
    % in the layer to a given input
    %
    % INPUTS:
    %   sig (num_in x T matrix): input
    %
    % OUTPUTS:
    %   sig (num_in x T matrix): input passed through activation function

    switch layer.act_func
        case 'lin'
        case 'relu'
            sig = max(0, sig);
        case 'sigmoid'
            sig = 1 ./ (1 + exp(-sig));
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of overflow - appx linear
            temp_sig(sig > layer.max_g) = sig(sig > layer.max_g);
            % take care of underflow so LL is defined (taking logs later)
            temp_sig(temp_sig < layer.min_pred_rate) = layer.min_pred_rate;
            sig = temp_sig;
    end
    
    end % method

    
    function sig = apply_act_deriv(layer, sig)
    % sig = layer.apply_act_deriv(sig)
    %
    % internal function that calculates the derivative of the activation 
    % function of the nodes in the layer to a given input
    %
    % INPUTS:
    %   sig (num_in x T matrix): input
    %
    % OUTPUTS:
    %   sig (num_in x T matrix): input passed through derivative of 
    %       activation function

    switch layer.act_func
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            if 1
                sig = relu_deriv_inplace(sig);
            else
                sig(sig <= 0) = 0; sig(sig > 0) = 1;
            end
        case 'sigmoid'
            % sig = exp(-sig) ./ (1 + exp(-sig)).^2;
            sig = 1 ./ (exp(-sig/2) + exp(sig/2)).^2; % no Infs
        case 'softplus'
            % temp_sig = exp(sig) ./ (1 + exp(sig));
            % % e^x/(1+e^x) => 1 for large x
            % temp_sig(sig > layer.max_g) = 1; 
            % sig = temp_sig;
            sig = 1 ./ (1 + exp(-sig)); % ~twice as fast
    end
    
    end % method
    
end
%% ********************  static methods ***********************************
methods (Static)
    
    function reg_lambdas = init_reg_lambdas()
    % reg_lambdas = Layer.init_reg_lambdas()
    %
    % creates reg_lambdas struct and sets default values to 0; called from
    % Layer constructor
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_lambdas (struct): contains initialized reg params

    reg_lambdas.l2_weights = 0;     % L2 on weights
    reg_lambdas.l2_biases = 0;      % L2 on bias params
	reg_lambdas.d2t_hid = 0;		% d2t on hidden layer activation
    
    end % method
 
    
    function [weights, biases, init_params] = set_init_weights_stat( ...
                                            init_method, num_in, num_out)
                                            
    % [weights, biases, init_params] = Layer.set_init_weights_stat( ...
    %                                       init_method, num_in, num_out)
    % 
    % static function that initializes weights/biases and sets init_params
    % structure based on input. Called from the Layer constructor and from
    % the non-static method set_init_weights
    %
    % INPUTS:
    %   init_method (string): specifies a random initialization; see
    %       allowed_init_types for supported options; or a 2x1 cell array
    %       containing weights and biases of appropriate dimensions
    %   num_in (scalar): number of input nodes
    %   num_out (scalar): number of output nodes
    %
    % OUTPUTS:
    %   weights (num_out x num_in matrix)
    %   biases (num_out x 1 vector)
    %   init_params (struct): specifies init_weights and rng_state
    
    if ischar(init_method)
        
        init_params.rng_state = rng();
        init_params.init_weights = lower(init_method);
        
        % randomly initialize weights; start biases off at 0
        s = 1/sqrt(num_in);
        switch lower(init_method)
            case 'gauss'
                weights = s * randn(num_out, num_in);
            case 'trunc_gauss'
                weights = s * randn(num_out, num_in);
                indxs = find(weights > 2*s | weights < -2*s);
                while ~isempty(indxs)
                    weights(indxs) = s * randn(length(indxs),1);
                    indxs = find(weights > 2*s | weights < -2*s);
                end
            case 'uniform'
                % choose weights uniformly from the interval [-r, r]
                r = 4 * sqrt(6) / sqrt(num_out + num_in + 1);   
                weights = rand(num_out, num_in) * 2 * r - r;
            case 'orth'
                temp = s * randn(num_out, num_in);
                if num_in >= num_out
                    [u, ~, ~] = svd(temp');
                    weights = u(:, 1:num_out)';
                else
                    weights = temp;
                end
            otherwise
                error('Invalid initialization string "%s"', init_method)
        end
        biases = zeros(num_out, 1);

    elseif iscell(init_method)
        
        % use 'init_weights' to initialize weights
        assert(size(init_method{1}) == [num_out, num_in], ...
            'weight matrix has improper dimensions');
        assert(size(init_method{2}) == [num_out, 1], ...
            'weight matrix has improper dimensions');

        % get parameters
        weights = init_method{1};
        biases = init_method{2};
        
        init_params.rng_state = NaN;
        init_params.init_weights = 'user-supplied';
        
    else
        warning('init_weights must be a string or a cell array')
    end
    
    end % method

end

end