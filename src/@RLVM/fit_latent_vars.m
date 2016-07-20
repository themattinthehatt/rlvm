function [net, w2, latent_vars] = fit_latent_vars(net, fs)
% net = net.fit_latent_vars(fitting_struct)
%
% fits latent states
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
        Z = sum(sum(fs.pop_activity));
    otherwise
        error('Invalid noise distribution')
end
T = size(fs.pop_activity, 1);
num_cells = net.num_cells;

% for auto
num_hid_nodes = net.auto_subunit.num_hid_nodes;
if ischar(fs.init_weights)
    if strcmp(fs.init_weights, 'model')
        w2 = net.auto_subunit.w2;
        b2 = net.auto_subunit.b2;
    elseif strcmp(fs.init_weights, 'gauss')
        w2 = 0.1 * randn(size(net.auto_subunit.w2));
        b2 = 0.1 * randn(size(net.auto_subunit.b2));
    else
        error('Incorrect init_weights string')
    end
elseif isvector(fs.init_weights)
    assert(length(fs.init_weights) == ...
        num_cells + num_cells * num_hid_nodes, ...
        'init_weight vector does not have proper length')
else
    error('Improper init_weight format')
end
offsets = net.offsets;
lambda_sm  = net.auto_subunit.reg_lambdas.d2t_hid;

if lambda_sm > 0
    % dt
% 	reg_mat = spdiags([-1*ones(T,1) ones(T,1)],[0 1], T, T);
    % d2t
    reg_mat = spdiags([ones(T,1) -2*ones(T,1) ones(T,1)], [-1 0 1], T, T);
	reg_mat_grad = reg_mat'*reg_mat;
end

% ************************** RESHAPE WEIGHTS ******************************

if ischar(fs.init_latent_vars)
    if strcmp(fs.init_latent_vars, 'model')
        [~, gint] = net.auto_subunit.get_model_internals(fs.pop_activity); 
        init_params = gint{1}(:);
    elseif strcmp(fs.init_latent_vars, 'gauss')
        init_params = abs(randn(T * num_hid_nodes, 1));
    else
        error('Incorrect init_latent_vars string')
    end
elseif ismatrix(fs.init_latent_vars)
    assert(size(fs.init_latent_vars) == [T, num_hid_nodes], ...
        'init_latent_vars matrix does not have proper size')
    init_params = fs.init_latent_vars(:);
else
    error('Improper init_latent_vars format')
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
        [latent_vars, f, ~, output] = minFunc(obj_fun, init_params, ...
                                        optim_params);
    case 'fminunc'
        [latent_vars, f, ~, output] = fminunc(obj_fun, init_params, ...
                                        optim_params);
	case 'con'
		[latent_vars, f, ~, output] = minConf_SPG(obj_fun, init_params, ...
                                        @(t,b) max(t,0), optim_params);
end

[~, grad_pen] = objective_fun(latent_vars);
first_order_optim = max(abs(grad_pen));
if first_order_optim > 1e-2
    warning('First-order optimality: %.3f, fit might not be converged!', ...
        first_order_optim);
end

% ************************** UPDATE WEIGHTS *******************************
latent_vars = reshape(latent_vars, T, []);
net.auto_subunit.latent_vars = net.auto_subunit.apply_act_func(latent_vars);

% ************************** UPDATE HISTORY *******************************
curr_fit_details = struct( ...
    'fit_auto', 0, ...
    'fit_stim_individual', 0, ...
    'fit_stim_shared', 0, ...
    'fit_stim_subunits', 0, ...
    'fit_overall_offsets', 0, ...
    'fit_latent_vars', 1, ...
    'fit_weights', 0, ...
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
	lat_vars = reshape(params, T, num_hid_nodes);
    
    % ******************* COMPUTE FUNCTION VALUE **************************
    hidden_act = net.auto_subunit.apply_act_func(lat_vars);
    gint2 = bsxfun(@plus, hidden_act * w2, b2');
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
    end
    
    % ******************* COMPUTE GRADIENT ********************************
    % grad wrt latent_vars
    lat_var_grad = ((net.apply_spk_NL_deriv(gint2).*cost_grad)*w2') .* ...
                     net.auto_subunit.apply_act_deriv(lat_vars);
        
    % ******************* COMPUTE REG VALUES AND GRADIENTS ****************
    % smoothness penalty eval on hidden layer
    if lambda_sm > 0
        smooth_func = 0.5*sum(sum((reg_mat*lat_vars).^2)) / T;
        smooth_grad = reg_mat_grad*lat_vars / T;
    else
        smooth_func = 0;
        smooth_grad = zeros(size(hidden_act));
    end
    

    % ******************* COMBINE *****************************************				
    cost_func = cost_func / Z + lambda_sm * smooth_func;
    grad = lat_var_grad(:) / Z + lambda_sm * smooth_grad(:);
%     fprintf('cost_func: %g\n', cost_func / Z)
%     fprintf('reg_pen: %g\n\n', lambda_sm * smooth_func)
    end % internal function

end % fit_auto method
