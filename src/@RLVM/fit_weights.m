function [net, weights, latent_vars] = fit_weights(net, fs)
% net = net.fit_weights(fitting_struct)
%
% fits weights from latent states to cells
%
% INPUTS:
%   fs:                 struct created in method RLVM.fit_model
%       pop_activity:   T x num_cells matrix of population activity
%       Xstims:         cell array holding T x _ matrices of stim info
%       fit_subs:       vector of scalar indices for subunits (not used)
%       init_weights
%       init_latent_vars
%
% OUTPUTS:
%   net:                updated RLVM object

% ************************** CHECK INPUTS *********************************

% ensure size of pop_activity is consistent with autoencoder weights
net.auto_subunit.check_inputs(fs.pop_activity);

% ************************** DEFINE USEFUL QUANTITIES *********************
switch net.noise_dist
    case 'gauss'
        Z = numel(fs.pop_activity);
    case 'poiss'
        Z = sum(fs.pop_activity(:));
    case 'bern'
        Z = numel(fs.pop_activity);
    otherwise
        error('Invalid noise distribution')
end
T = size(fs.pop_activity, 1);
num_cells = net.num_cells;

% for auto
num_hid_nodes = net.auto_subunit.num_hid_nodes;
lambda_w2  = net.auto_subunit.reg_lambdas.l2_weights2;
lambda_b2 = net.auto_subunit.reg_lambdas.l2_biases2;
if ischar(fs.init_latent_vars)
    if strcmp(fs.init_latent_vars, 'model')
        [~, gint] = net.auto_subunit.get_model_internals(fs.pop_activity); 
        assert(all(size(gint{1}) == [T, num_hid_nodes]), ...
            'size mismatch between stored latent_vars matrix and input')
        latent_vars = gint{1};
    elseif strcmp(fs.init_latent_vars, 'gauss')
        latent_vars = abs(randn(T, num_hid_nodes));
    else
        error('Incorrect init_latent_vars string')
    end
elseif ismatrix(fs.init_latent_vars)
    assert(all(size(fs.init_latent_vars) == [T, num_hid_nodes]), ...
        'init_latent_vars matrix does not have proper size')
    latent_vars = fs.init_latent_vars;
else
    error('Improper init_latent_vars format')
end
offsets = net.offsets;

% ************************** RESHAPE WEIGHTS ******************************

if ischar(fs.init_weights)
    if strcmp(fs.init_weights, 'model')
        init_params = [net.auto_subunit.w2(:); net.auto_subunit.b2(:)];
    elseif strcmp(fs.init_weights, 'gauss')
        init_params = 0.1 * randn(num_cells + num_cells * num_hid_nodes, 1);
    else
        error('Incorrect init_weights string')
    end
elseif isvector(fs.init_weights)
    assert(length(fs.init_weights) == ...
        net.num_cells + net.num_cells * num_hid_nodes, ...
        'init_weight vector does not have proper length')
    init_params = fs.init_weights;
else
    error('Improper init_weight format')
end

% ************************** FIT MODEL ************************************
optim_params = net.optim_params;
if net.fit_params.deriv_check
    optim_params.Algorithm = 'quasi-newton';
    optim_params.HessUpdate = 'steepdesc';
    optim_params.GradObj = 'on';
    optim_params.DerivativeCheck = 'on';
    optim_params.optimizer = 'fminunc';
    optim_params.FinDiffType = 'central';
    optim_params.maxIter = 0;
end

% define function handle to pass to optimizer
obj_fun = @(x) objective_fun(x);

% run optimization
if strcmp(optim_params.optimizer, 'minFunc') && ~exist('minFunc', 'file')
    optim_params.optimizer = 'fminunc';
end
switch optim_params.optimizer
    case 'minFunc'
        [weights, f, ~, output] = minFunc(obj_fun, init_params, ...
                                          optim_params);
    case 'fminunc'
        [weights, f, ~, output] = fminunc(obj_fun, init_params, ...
                                          optim_params);
	case 'con'
        [weights, f, ~, output] = minConf_SPG(obj_fun, init_params, ...
                                          @(t,b) max(t,0), optim_params);
end

[~, grad_pen] = objective_fun(weights);
first_order_optim = max(abs(grad_pen));
if first_order_optim > 1e-2
    warning('First-order optimality: %.3f, fit might not be converged!', ...
        first_order_optim);
end

% ************************** UPDATE WEIGHTS *******************************

[net.auto_subunit.w2, net.auto_subunit.b2] = ...
                net.auto_subunit.get_decoding_weights(weights);

% ************************** UPDATE HISTORY *******************************
curr_fit_details = struct( ...
    'fit_auto', 0, ...
    'fit_stim_individual', 0, ...
    'fit_stim_shared', 0, ...
    'fit_stim_subunits', 0, ...
    'fit_overall_offsets', 0, ...
    'fit_latent_vars', 0, ...
    'fit_weights', 1, ...
    'func_val', f, ...
    'first_order_opt', output.firstorderopt, ...
    'exit_msg',output.message);
net.fit_history = cat(1, net.fit_history, curr_fit_details);


    %% ******************** nested objective function *********************
    function [cost_func, grad] = objective_fun(params)
    % function for calculating the mean square cost function and
    % the gradient with respect to the weights and biases for the 2
    % layer autoencoder network.
    % INPUTS:
    % OUTPUTS:

    % ******************* INITIALIZATION **********************************
    
    % ******************* PARSE PARAMETER VECTOR **************************
	[w2, b2] = net.auto_subunit.get_decoding_weights(params);
    
    % ******************* COMPUTE FUNCTION VALUE **************************
    gint2 = bsxfun(@plus, latent_vars * w2, b2');
    pred_activity = bsxfun(@plus, net.apply_spk_NL(gint2), offsets');
	
    % cost function and gradient eval wrt predicted output
    switch net.noise_dist
        case 'gauss'
            cost_grad = pred_activity - fs.pop_activity;
            cost_func = 0.5*sum(sum(cost_grad.^2));
        case 'poiss'
            % calculate cost function
            cost_func = -sum(sum(fs.pop_activity.*log(pred_activity) - pred_activity));
            % calculate gradient
            cost_grad = -(fs.pop_activity./pred_activity - 1);
            % set gradient equal to zero where underflow occurs
            cost_grad(pred_activity <= net.min_pred_rate) = 0;
        case 'bern'
            % calculate cost function
            cost_func1 = fs.pop_activity.*log(a{end});
            cost_func1(a{end}==0) = 0;
            cost_func2 = (1-fs.pop_activity).*log(1-a{end});
            cost_func2(a{end}==1) = 1;
            cost_func = -sum(sum(cost_func1 + cost_func2));
            % calculate gradient
            cost_grad = -(fs.pop_activity./a{end} - ...
                         (1-fs.pop_activity)./(1-a{end}));
            % set gradient equal to zero where underflow occurs
            cost_grad(a{end} <= net.min_pred_rate) = 0;
            cost_grad(a{end} >= 1-net.min_pred_rate) = 0;
    end
    
    % ******************* COMPUTE GRADIENT ********************************
    gb2 = net.apply_spk_NL_deriv(gint2) .* cost_grad;
    gw2 = latent_vars' * gb2;
    weight_grad = [gw2(:); sum(gb2,1)'];
        
    % ******************* COMPUTE REG VALUES AND GRADIENTS ****************
    reg_pen = 0.5 * lambda_w2 * sum(sum(w2.^2)) ...
            + 0.5 * lambda_b2 * sum(b2.^2);
    
    reg_pen_grad = [lambda_w2 * w2(:); lambda_b2 * b2(:)];
    
    % ******************* COMBINE *****************************************				
    cost_func = cost_func / Z + reg_pen;
    grad = weight_grad(:) / Z + reg_pen_grad;

    end % internal function

end % fit_auto method
