function net = fit_weights_latent_vars(net, fs)
% net = net.weights_latent_vars(fs)
%
% Simultaneously fits model parameters - both autoencoder weights and 
% stimulus filters
%
% INPUTS:
%   fs:                 struct created in method RLVM.fit_model
%       pop_activity:   num_cells x T matrix of population activity
%       Xstims:         cell array holding T x _ matrices of stim info
%
% OUTPUTS:
%   net:                updated RLVM object
%
% TODO:
%   non_targ_sig does not work for selectively fitting shared StimSubunits

% ************************** DEFINE USEFUL QUANTITIES *********************
switch net.fit_params.noise_dist
    case 'gauss'
        Z = numel(fs.pop_activity);
    case 'poiss'
        Z = sum(fs.pop_activity(:));
    case 'bern'
        Z = numel(fs.pop_activity);
    otherwise
        error('Invalid noise distribution')
end
T = size(fs.pop_activity,2);
num_cells = size(net.layers(end).weights,1);
num_layers = length(net.layers);

if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
    % fit all stim subunit outputs
    num_subs = length(net.stim_subunits);
    x_targets = [net.stim_subunits(:).x_target];
    mod_signs = [net.stim_subunits(:).mod_sign];
elseif ~isempty(net.stim_subunits)
    % don't fit stim subunit outputs but use for model evaluation
    num_subs = 0;
    if net.fit_params.fit_stim_shared
        error('Holding weights fixed for shared subunits not supported')
    end
else
    num_subs = [];
end

if net.fit_params.fit_stim_shared
    int_layer = net.integration_layer;
    % decide if we can speed up function/gradient evaluations
    if strcmp([net.stim_subunits(:).NL_type], ...
            repmat(net.stim_subunits(1).NL_type, 1, num_subs)) ...
            && length(unique(mod_signs)) == 1 ...
            && length(unique(x_targets)) == 1
        use_batch_calc = 1;
    else
        use_batch_calc = 0;
    end
else
    int_layer = 0;
end

% ************************** RESHAPE WEIGHTS ******************************
init_params = [];
param_tot = 0;

% store length of each layer's weight/bias vector
layer_lens = zeros(num_layers, 2);
% store index values of each layer's weight/bias vector
layer_indxs_full = cell(num_layers, 2); % index into full param vec
layer_indxs = cell(num_layers, 2);      % index into layer param vec
num_layer_indxs = 0;
for i = 1:num_layers
    curr_params = [net.layers(i).weights(:); net.layers(i).biases(:)];
    % add current params to initial param vector
    init_params = [init_params; curr_params];
    % length of layer vector
    layer_lens(i,1) = length(net.layers(i).weights(:));
    layer_lens(i,2) = length(net.layers(i).biases(:));
    % param indices into full param vector
    if net.fit_params.fit_auto
        layer_indxs_full{i,1} = param_tot + (1:layer_lens(i,1));
        param_tot = param_tot + layer_lens(i, 1);
    else
        layer_indxs_full{i,1} = [];
    end
    layer_indxs_full{i,2} = param_tot + (1:layer_lens(i,2));
    param_tot = param_tot + layer_lens(i, 2);
    % param indices into param vector just assoc'd w/ layer params
    if net.fit_params.fit_auto
        layer_indxs{i,1} = num_layer_indxs + (1:layer_lens(i,1));
        num_layer_indxs = num_layer_indxs + layer_lens(i,1);
    else
        layer_indxs{i,1} = [];
    end
    layer_indxs{i,2} = num_layer_indxs + (1:layer_lens(i,2));
    num_layer_indxs = num_layer_indxs + layer_lens(i,2);
end    

if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
    % store length of each (target) sub's filter
    filt_lens = zeros(num_subs, 1);
    % store index values of each subunit's filter coeffs
    stim_indxs_full = cell(num_subs, 1);  % index into full param vec
    stim_indxs = cell(num_subs, 1);       % index into stim param vec
    num_stim_indxs = 0;
    for i = 1:num_subs
        curr_params = net.stim_subunits(i).filt(:);
        % add current params to initial param vector
        init_params = [init_params; curr_params]; 
        % length of filter
        filt_lens(i) = length(curr_params);
        % param indices into full param vector
        stim_indxs_full{i} = param_tot + (1:filt_lens(i));
        param_tot = param_tot + filt_lens(i);
        % param indices into param vector just assoc'd w/ subunit's filters
        stim_indxs{i} = num_stim_indxs + (1:filt_lens(i));
        num_stim_indxs = num_stim_indxs + filt_lens(i);
    end
else
    stim_indxs = [];
    num_stim_indxs = 0;
end

if net.fit_params.fit_stim_shared
    curr_params = net.layers(int_layer).ext_weights(:);
    init_params = [init_params; curr_params];
    num_stim_weights_indxs = length(curr_params);
    stim_weights_indxs = param_tot + (1:num_stim_weights_indxs);
    param_tot = param_tot + num_stim_weights_indxs;
end
    
% keep track of output from model components that are not being fit
if num_subs == 0
    % loop over the subunits and compute the generating signals
    non_targ_sig = zeros(size(fs.pop_activity));
    for i = 1:length(net.stim_subunits)
        temp_sub = net.stim_subunits(i);
        temp_g = fs.Xstims{temp_sub.x_target} * temp_sub.filt;
        temp_f = temp_sub.apply_NL_func(temp_g);
        non_targ_sig = non_targ_sig + (temp_sub.mod_sign * temp_f)';
    end
else
    non_targ_sig = 0;
end
    
% ************************** FIT MODEL ************************************
optim_params = net.optim_params;
if net.optim_params.deriv_check
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
        [params, f, ~, output] = minFunc(obj_fun, init_params, ...
                                          optim_params);
    case 'fminunc'
        [params, f, ~, output] = fminunc(obj_fun, init_params, ...
                                          optim_params);
  	case 'con'
        [params, f, ~, output] = minConf_SPG(obj_fun, init_params, ...
                                          @(t,b) max(t,0), optim_params);
end

[~, grad] = objective_fun(params);
first_order_optim = max(abs(grad));
if first_order_optim > 1e-2
    warning('First-order optimality: %.3f, fit might not be converged!', ...
        first_order_optim);
end

% ************************** UPDATE WEIGHTS *******************************
for i = 1:num_layers
    curr_params = params(layer_indxs_full{i,1});
    net.layers(i).weights = reshape(curr_params, size(net.layers(i).weights));
    net.layers(i).biases = params(layer_indxs_full{i,2});
end
if net.fit_params.fit_stim_individual
    for i = 1:num_subs
        curr_params = params(stim_indxs_full{i});
        net.stim_subunits(i).filt = ...
            reshape(curr_params, [], num_cells);
    end
elseif net.fit_params.fit_stim_shared
    for i = 1:num_subs
        net.stim_subunits(i).filt = params(stim_indxs_full{i});
    end
    curr_params = params(stim_weights_indxs);
    net.layers(int_layer).ext_weights = reshape(curr_params, ...
            size(net.layers(int_layer).ext_weights));
end

% ************************** UPDATE HISTORY *******************************
curr_fit_details = struct( ...
    'fit_auto', net.fit_params.fit_auto, ...
    'fit_stim_individual', net.fit_params.fit_stim_individual, ...
    'fit_stim_shared', net.fit_params.fit_stim_shared, ...
    'fit_latent_vars', 0, ...
    'fit_weights', 0, ...
    'func_val', f, ...
    'func_vals', output.trace.fval, ...
    'iters', output.iterations, ...
    'func_evals', output.funcCount, ...
    'first_order_opt', output.firstorderopt, ...
    'exit_msg',output.message);
net.fit_history = cat(1, net.fit_history, curr_fit_details);


    %% ******************** nested objective function *********************
    function [func, grad] = objective_fun(params)
    % Calculates the loss function and its gradient with respect to the 
    % model parameters

    % ******************* INITIALIZATION **********************************
    z = cell(num_layers,1);
    a = cell(num_layers,1);
    weights = cell(num_layers,1);
    biases = cell(num_layers,1);
    grad_weights = cell(num_layers,1);
    grad_biases = cell(num_layers,1);
    if net.fit_params.fit_stim_shared
        gint = zeros(T,num_subs);   % store filter outputs
        fgint = gint;               % store subunit outputs
        filts = cell(num_subs,1);   % filters for all (target) subs
    elseif net.fit_params.fit_stim_individual
        gint = cell(num_subs,1);   % store filter outputs
        fgint = gint;               % store subunit outputs
        filts = cell(num_subs,1);   % filters for all (target) subs        
    end
    % Note: cell arrays do not need to be stored contiguously 
    
    % ******************* PARSE PARAMETER VECTOR **************************
    for ii = 1:num_layers
        weights{ii} = reshape(params(layer_indxs_full{ii,1}), ...
                              size(net.layers(ii).weights));
        biases{ii} = params(layer_indxs_full{ii,2});
    end
    if net.fit_params.fit_stim_individual
        for ii = 1:num_subs 
            filts{ii} = reshape(params(stim_indxs_full{ii}), [], num_cells);
        end
    elseif net.fit_params.fit_stim_shared
        for ii = 1:num_subs
            filts{ii} = params(stim_indxs_full{ii});
        end
        stim_weights = reshape(params(stim_weights_indxs), ...
                     size(net.layers(int_layer).ext_weights));
    end
    
    % ******************* COMPUTE FUNCTION VALUE **************************
    if net.fit_params.fit_stim_individual
        % loop over the subunits and compute the generating signals
        for ii = 1:num_subs 
            gint{ii} = fs.Xstims{x_targets(ii)} * filts{ii};
            fgint{ii} = net.stim_subunits(ii).apply_NL_func(gint{ii});
        end
    elseif net.fit_params.fit_stim_shared
        if use_batch_calc
            gint = fs.Xstims{x_targets(1)} * [filts{:}];
            fgint = net.stim_subunits(1).apply_NL_func(gint);
        else
            % loop over the subunits and compute the generating signals
            for ii = 1:num_subs 
                gint(:,ii) = fs.Xstims{x_targets(ii)} * filts{ii};
                fgint(:,ii) = net.stim_subunits(ii).apply_NL_func(gint(:,ii));
            end
        end
    end
    for ii = 1:num_layers
        if ii == 1
            if ~isempty(weights{1})
                % auto model
                z{ii} = bsxfun(@plus, weights{ii}*fs.pop_activity, ...
                                      biases{ii});
            else
                % just stimulus model
                z{ii} = repmat(biases{ii}, 1, T);
            end
        else
            z{ii} = bsxfun(@plus, weights{ii}*a{ii-1}, ...
                                  biases{ii});
        end
        if ii == int_layer
            z{ii} = z{ii} + stim_weights * fgint';
        end
        if ii == num_layers
            if net.fit_params.fit_stim_individual
                for jj = 1:num_subs
                    z{ii} = z{ii} + (mod_signs(jj) * fgint{jj})';
                end
            end
            % add contribution from part of model not being fit
            z{ii} = z{ii} + non_targ_sig;
        end
        a{ii} = net.layers(ii).apply_act_func(z{ii});
    end
    % cost function and gradient eval wrt predicted output
    switch net.fit_params.noise_dist
        case 'gauss'
            cost_grad = (a{end} - fs.pop_activity);
            cost_func = 0.5*sum(sum(cost_grad.^2));
        case 'poiss'
            % calculate cost function
            cost_func = -sum(sum(fs.pop_activity.*log(a{end}) - a{end}));
            % calculate gradient
            cost_grad = -(fs.pop_activity./a{end} - 1);
            % set gradient equal to zero where underflow occurs
            cost_grad(a{end} <= net.min_pred_rate) = 0;
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
    
    % ******************* COMPUTE GRADIENTS *******************************
    % backward pass, last layer
    delta = net.layers(end).apply_act_deriv(z{end}) .* cost_grad;
    if net.fit_params.fit_stim_individual
        stim_delta = delta;
    end
    if num_layers > 1
        % only perform if a{end-1} exists (use fs.pop_activity otherwise)
        grad_weights{end} = delta * a{end-1}';
        grad_biases{end} = sum(delta, 2);
    end
    if int_layer == num_layers
        int_delta = delta;
    end
    % backward pass, hidden layers
    for ii = (num_layers-1):-1:2
        temp1 = net.layers(ii).apply_act_deriv(z{ii});
        temp2 = (weights{ii+1}' * delta);
        delta = temp1 .* temp2;
        grad_weights{ii} = delta * a{ii-1}';
        grad_biases{ii} = sum(delta, 2);
        if ii == int_layer
            % save delta value for shared stim subunit gradient
            int_delta = delta;
        end
    end
    % backward pass, first hidden layer
    if num_layers > 1
        delta = net.layers(1).apply_act_deriv(z{1}) .* ...
                (weights{2}' * delta);
        % else use delta calculated above
    end
    if ~isempty(net.layers(1).weights)
        grad_weights{1} = delta * fs.pop_activity';
        grad_biases{1} = sum(delta, 2);
    else
        grad_weights{1} = [];
        grad_biases{1} = sum(delta, 2);
    end
    if int_layer == 1
        int_delta = delta;
    end
        
    % construct gradient vector
    layer_grad = [];
    for ii = 1:num_layers
        layer_grad = [layer_grad; grad_weights{ii}(:); grad_biases{ii}];
    end
    
    if net.fit_params.fit_stim_individual
        % initialize LL gradient
        stim_grad = zeros(num_stim_indxs, 1);
        for ii = 1:num_subs 
            delta = stim_delta .* ...
                    net.stim_subunits(ii).apply_NL_deriv(gint{ii})';
            temp = (mod_signs(ii) * delta * fs.Xstims{x_targets(ii)})';
            stim_grad(stim_indxs{ii}) = temp(:);
        end
        stim_weights_grad = [];
    elseif net.fit_params.fit_stim_shared
        
        stim_weights_grad = int_delta * fgint;
       
        if use_batch_calc
            % only when all subunits are same; empirically, this method is
            % ~X% faster with T = 16000, num_cells = 375, num_subs = 10           
            delta = net.stim_subunits(1).apply_NL_deriv(gint)' .* ...
                (stim_weights' * int_delta);
            stim_grad = (delta * fs.Xstims{1})';
        else
%             for ii = 1:num_subs 
%                 delta = net.stim_subunits(ii).apply_NL_deriv(gint(:,ii)' .* ...
%                         (stim_weights(ii,:) * int_delta);
%                 stim_grad(stim_indxs{ii}) = sum( ...
%                     (fs.Xstims{x_targets(ii)})' * deriv * mod_signs(ii), 2);
%             end
        end
    else
        stim_grad = [];
        stim_weights_grad = [];
    end

    
    % ******************* COMPUTE REG VALUES AND GRADIENTS ****************

    layer_reg_pen = 0;
    layer_reg_pen_grad = zeros(num_layer_indxs,1);
    for ii = 1:num_layers
        % get reg pen for weights
        if ~isempty(weights{ii})
            reg_lambda = net.layers(ii).reg_lambdas.l2_weights;
            layer_reg_pen = layer_reg_pen + 0.5 * reg_lambda * ...
                            sum(weights{ii}(:).^2);
            layer_reg_pen_grad(layer_indxs{ii,1}) = reg_lambda * ...
                                                    weights{ii}(:);
        end
        % get reg pen for biases
        reg_lambda = net.layers(ii).reg_lambdas.l2_biases;
        layer_reg_pen = layer_reg_pen + 0.5 * reg_lambda * sum(biases{ii}.^2);
        layer_reg_pen_grad(layer_indxs{ii,2}) = reg_lambda * biases{ii};
    end
    layer_grad = layer_grad / Z + layer_reg_pen_grad;

    if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
        stim_reg_pen = zeros(num_subs,1);
        stim_reg_pen_grad = zeros(num_stim_indxs,1);
        for ii = 1:num_subs
            stim_reg_pen(ii) = net.stim_subunits(ii).reg_lambdas.l2;
            stim_reg_pen_grad(stim_indxs{ii}) = stim_reg_pen(ii)*filts{ii}(:);
        end
        stim_reg_pen = 0.5*stim_reg_pen'*cellfun(@(x) sum(sum(x.^2)), filts);
        stim_grad = stim_grad(:) / Z + stim_reg_pen_grad;
    else
        stim_reg_pen = 0;
    end
    
%     if net.fit_params.fit_stim_shared
%         stim_weights_reg_pen = 0.5 * net.lambda_stim * sum(sum(stim_weights.^2));
%         stim_weights_reg_pen_grad = net.lambda_stim * stim_weights(:);
%         stim_weights_grad = stim_weights_grad(:) / Z + stim_weights_reg_pen_grad;
%     else
%         stim_weights_reg_pen = 0;
%     end
    stim_weights_reg_pen = 0;
  
    
    % ******************* COMBINE *****************************************
    
    func = cost_func / Z + layer_reg_pen ...
                         + stim_reg_pen ...
                         + stim_weights_reg_pen;
                     
    grad = [layer_grad; stim_grad; stim_weights_grad(:) / Z];
    
    end % internal function

end % fit_weights method




