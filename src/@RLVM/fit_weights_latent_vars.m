function net = fit_weights_latent_vars(net, fs)
% net = net.weights_latent_vars(fitting_struct)
%
% Simultaneously fits model parameters, both autoencoder weights and 
% stimulus filters
%
% INPUTS:
%   fs:                 struct created in method RLVM.fit_model
%       pop_activity:   T x num_cells matrix of population activity
%       Xstims:         cell array holding T x _ matrices of stim info
%       fit_subs:       vector of scalar indices for subunits
%
% OUTPUTS:
%   net:                updated RLVM object
%
% TODO:
%   nontarg_g does not work for selectively fitting shared StimSubunits
%   (need to use/fit only part of stim_weights)

% ************************** CHECK INPUTS *********************************
if net.fit_params.fit_auto
    % ensure size of pop_activity is consistent with autoencoder weights
    net.auto_subunit.check_inputs(fs.pop_activity);
end
if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
    % ensure size of Xstims is consistent with subunit filters
    fit_subs = fs.fit_subs;             % indices of subunits to fit
    num_fit_subs = length(fit_subs);    % number of targeted subunits
    for i = 1:num_fit_subs
        net.stim_subunits(fit_subs(i)).check_inputs(fs.Xstims);
    end
end

% ************************** DEFINE USEFUL QUANTITIES *********************
switch net.noise_dist
    case 'gauss'
        Z = numel(fs.pop_activity);
    case 'poiss'
        Z = sum(sum(fs.pop_activity));
    otherwise
        error('Invalid noise distribution')
end
T = size(fs.pop_activity,1);
num_cells = net.num_cells;
lambda_off = net.lambda_off;

if net.fit_params.fit_auto
    num_hid_nodes = net.auto_subunit.num_hid_nodes;
    lambda_w1  = net.auto_subunit.reg_lambdas.l2_weights1;
    lambda_w2  = net.auto_subunit.reg_lambdas.l2_weights2;
    lambda_b1 = net.auto_subunit.reg_lambdas.l2_biases1;
    lambda_b2 = net.auto_subunit.reg_lambdas.l2_biases2;
    lambda_sp = net.auto_subunit.reg_lambdas.l1_hid;
end

if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
    num_subs = length(net.stim_subunits);
    non_fit_subs = setdiff(1:num_subs, fit_subs);
    x_targets = [net.stim_subunits(fit_subs).x_target];
    mod_signs = [net.stim_subunits(fit_subs).mod_sign];
elseif ~isempty(net.stim_subunits)
    % use all stim subunit outputs for model evaluation
    fit_subs = [];
    non_fit_subs = 1:length(net.stim_subunits);
else
    fit_subs = [];
    non_fit_subs = [];
end

if net.fit_params.fit_stim_shared
    % decide if we can speed up function/gradient evaluations
    if strcmp([net.stim_subunits(:).NL_type], ...
            repmat(net.stim_subunits(1).NL_type, 1, num_subs)) ...
            && length(unique([net.stim_subunits(:).mod_sign])) == 1 ...
            && length(unique([net.stim_subunits(:).x_target])) == 1
        use_batch_calc = 1;
    else
        use_batch_calc = 0;
    end
end

% ************************** RESHAPE WEIGHTS ******************************
init_params = [];
param_tot = 0;

if net.fit_params.fit_auto
    if net.auto_subunit.weight_tie
        init_params = [init_params; ...
                       net.auto_subunit.w2(:); ...
                       net.auto_subunit.b1; net.auto_subunit.b2];
        num_weights = num_cells * num_hid_nodes + num_cells + num_hid_nodes;       
    else
        init_params = [init_params; ...
                       net.auto_subunit.w1(:); net.auto_subunit.w2(:); ...
                       net.auto_subunit.b1; net.auto_subunit.b2];
        num_weights = 2 * num_cells * num_hid_nodes + num_cells + num_hid_nodes;
    end
    auto_indxs = param_tot + (1:num_weights);
    num_auto_indxs = length(auto_indxs);
    param_tot = param_tot + num_auto_indxs;
else
    auto_indxs = [];
    num_auto_indxs = 0;
end

if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
    % store length of each (target) sub's filter
    filt_lens = zeros(num_fit_subs, 1);
    % store index values of each subunit's filter coeffs
    stim_indxs_full = cell(num_fit_subs, 1); 
    stim_indxs = cell(num_fit_subs, 1); 
    num_stim_indxs = 0;
    for i = 1:num_fit_subs 
        curr_filt = net.stim_subunits(fit_subs(i)).filt(:);
        % add current coeffs to initial param vector
        init_params = [init_params; curr_filt]; 
        % length of filter
        filt_lens(i) = length(curr_filt);
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
    init_params = [init_params; net.stim_weights(:)];
    num_stim_weights_indxs = length(net.stim_weights(:));
    stim_weights_indxs = param_tot + (1:num_stim_weights_indxs);
    param_tot = param_tot + num_stim_weights_indxs;
end

% add constant offset to capture mean for each cell
if net.fit_params.fit_overall_offsets
    init_params = [init_params; net.offsets];
    offset_indxs = param_tot + (1:net.num_cells);
    num_offset_indxs = net.num_cells;
    param_tot = param_tot + num_offset_indxs;
else
    offset_indxs = [];
    num_offset_indxs = 0;
end
    
% keep track of output from model components that are not being fit
nontarg_g = zeros(T, net.num_cells);
if ~isempty(net.auto_subunit) && ~net.fit_params.fit_auto
    [~, auto_gint] = net.auto_subunit.get_model_internals(fs.pop_activity);
    nontarg_g = nontarg_g + auto_gint{2};
end
for i = non_fit_subs
    nontarg_g = nontarg_g + net.stim_subunits(i).mod_sign * ...
                       net.stim_subunits(i).get_model_internals(fs.Xstims);
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
        [params, f, ~, output] = minFunc(obj_fun, init_params, ...
                                          optim_params);
    case 'fminunc'
        [params, f, ~, output] = fminunc(obj_fun, init_params, ...
                                          optim_params);
  	case 'con'
        [params, f, ~, output] = minConf_SPG(obj_fun, init_params, ...
                                          @(t,b) max(t,0), optim_params);
end

[~, grad_pen] = objective_fun(params);
first_order_optim = max(abs(grad_pen));
if first_order_optim > 1e-2
    warning('First-order optimality: %.3f, fit might not be converged!', ...
        first_order_optim);
end

% ************************** UPDATE WEIGHTS *******************************
if net.fit_params.fit_auto
    [net.auto_subunit.w1, net.auto_subunit.w2, ...
     net.auto_subunit.b1, net.auto_subunit.b2] = ...
     net.auto_subunit.get_weights(params(auto_indxs));
 
    net.auto_subunit = net.auto_subunit.set_hidden_order(fs.pop_activity);
end
if net.fit_params.fit_stim_individual
    for i = 1:num_fit_subs
        curr_filt = params(stim_indxs_full{i});
        net.stim_subunits(fit_subs(i)).filt = ...
            reshape(curr_filt, [], num_cells);
    end
elseif net.fit_params.fit_stim_shared
    for i = 1:num_fit_subs
        net.stim_subunits(fit_subs(i)).filt = params(stim_indxs_full{i});
    end
    net.stim_weights = reshape(params(stim_weights_indxs), [], num_cells);
end
if net.fit_params.fit_overall_offsets
    net.offsets = params(offset_indxs);
end

% ************************** UPDATE HISTORY *******************************
curr_fit_details = struct( ...
    'fit_auto', net.fit_params.fit_auto, ...
    'fit_stim_individual', net.fit_params.fit_stim_individual, ...
    'fit_stim_shared', net.fit_params.fit_stim_shared, ...
    'fit_stim_subunits', fit_subs, ...
    'fit_overall_offsets', net.fit_params.fit_overall_offsets, ...
    'fit_latent_vars', 0, ...
    'fit_weights', 0, ...
    'func_val', f, ...
    'first_order_opt', output.firstorderopt, ...
    'exit_msg',output.message);
net.fit_history = cat(1, net.fit_history, curr_fit_details);


    %% ******************** nested objective function *********************
    function [func, grad] = objective_fun(params)
    % Calculates the loss function and its gradient with respect to the 
    % model parameters

    % ******************* INITIALIZATION **********************************
    if net.fit_params.fit_stim_individual
        gint = cell(num_fit_subs, 1);   % store filter outputs
        fgint = gint;                   % store subunit outputs
        filts = cell(num_fit_subs, 1);  % filters for all (target) subs
    elseif net.fit_params.fit_stim_shared
        gint = zeros(T, num_fit_subs);  % store filter outputs
        fgint = gint;                   % store subunit outputs
        filts = cell(num_fit_subs, 1);  % filters for all (target) subs
    end
    % Note: cell arrays do not need to be stored contiguously 
    
    % ******************* PARSE PARAMETER VECTOR **************************
    if net.fit_params.fit_auto
        [w1, w2, b1, b2] = net.auto_subunit.get_weights(params(auto_indxs));
    end
    if net.fit_params.fit_stim_individual
        for ii = 1:num_fit_subs 
            filts{ii} = reshape(params(stim_indxs_full{ii}), [], num_cells);
        end
    elseif net.fit_params.fit_stim_shared
        for ii = 1:num_fit_subs
            filts{ii} = params(stim_indxs_full{ii});
        end
        stim_weights = reshape(params(stim_weights_indxs), [], num_cells);
    end
    if net.fit_params.fit_overall_offsets
        offsets = params(offset_indxs);
    else
        offsets = zeros(num_cells, 1);
    end
    
    % ******************* COMPUTE FUNCTION VALUE **************************
    % initialize overall generating function G with the offset term and the
    % contribution from nontarget subs
    G = bsxfun(@plus, nontarg_g, offsets'); 
    
    if net.fit_params.fit_auto
        gint1 = bsxfun(@plus, fs.pop_activity * w1, b1');
        hidden_act = net.auto_subunit.apply_act_func(gint1);
        G = G + bsxfun(@plus, hidden_act * w2, b2');
    end
    if net.fit_params.fit_stim_individual
        % loop over the subunits and compute the generating signals
        for ii = 1:num_fit_subs 
            gint{ii} = fs.Xstims{x_targets(ii)} * filts{ii};
            fgint{ii} = net.stim_subunits(fit_subs(ii)).apply_NL_func(gint{ii});
            G = G + fgint{ii} * mod_signs(ii);
        end
    elseif net.fit_params.fit_stim_shared
        if use_batch_calc
            gint = fs.Xstims{x_targets(1)} * [filts{:}];
            fgint = net.stim_subunits(fit_subs(1)).apply_NL_func(gint);
        else
            % loop over the subunits and compute the generating signals
            for ii = 1:num_fit_subs 
                gint(:,ii) = fs.Xstims{x_targets(ii)} * filts{ii};
                fgint(:,ii) = net.stim_subunits(fit_subs(ii)).apply_NL_func(gint(:,ii));
            end
        end
        G = G + fgint * stim_weights;
    end
    % cost function and gradient eval wrt predicted output
    pred_activity = net.apply_spk_NL(G);
    switch net.noise_dist
        case 'gauss'
            cost_grad = (pred_activity - fs.pop_activity);
            cost_func = 0.5*sum(sum(cost_grad.^2));
        case 'poiss'
            % calculate cost function
            cost_func = -sum(sum(fs.pop_activity.*log(pred_activity) - pred_activity));
            % calculate gradient
            cost_grad = -(fs.pop_activity./pred_activity - 1);
            % set gradient equal to zero where underflow occurs
            cost_grad(pred_activity <= net.min_pred_rate) = 0;
    end
    
    % ******************* COMPUTE GRADIENTS *******************************
    if net.fit_params.fit_auto
        % sparsity penalty eval on hidden layer
        if lambda_sp > 0
            [sparse_func, sparse_grad] = net.auto_subunit.get_sparse_penalty(hidden_act);
        else
            sparse_func = 0;
            sparse_grad = zeros(size(hidden_act,2),1);
        end
        % grad wrt biases
        gb2 = net.apply_spk_NL_deriv(G) .* cost_grad;
        gb1 = net.auto_subunit.apply_act_deriv(gint1).* ...
                bsxfun(@plus, gb2 * w2', lambda_sp * sparse_grad');
        % grad wrt weights
        if net.auto_subunit.weight_tie
            gw1 = [];
            gw2 = hidden_act' * gb2 + gb1' * fs.pop_activity;
        else
            gw1 = fs.pop_activity' * gb1;
            gw2 = hidden_act' * gb2;
        end
        gb1 = sum(gb1,1)';
        gb2 = sum(gb2,1)';
        auto_grad = [gw1(:); gw2(:); gb1; gb2];
    else
        auto_grad = [];
    end
    
    if net.fit_params.fit_stim_individual
        % initialize LL gradient
        stim_grad = zeros(num_stim_indxs, 1);
        residual = net.apply_spk_NL_deriv(G).*cost_grad;
        for ii = 1:num_fit_subs 
            deriv = residual.* ...
                    net.stim_subunits(fit_subs(ii)).apply_NL_deriv(gint{ii});
            stim_grad(stim_indxs{ii}) = reshape( ...
                (fs.Xstims{x_targets(ii)})' * deriv * mod_signs(ii), ...
                [], 1);
        end
        stim_weights_grad = [];
    elseif net.fit_params.fit_stim_shared
        % initialize LL gradient
        stim_grad = zeros(num_stim_indxs, 1);
        residual = net.apply_spk_NL_deriv(G).*cost_grad;
        stim_weights_grad = fgint' * residual;
        if use_batch_calc
            % only when all subunits are same; empirically, this method is
            % ~20% faster with T = 16000, num_cells = 375, num_subs = 10
            stim_weights = reshape(stim_weights, 1, num_subs, num_cells);
            residual = reshape(residual, [], 1, num_cells);
            temp1 = bsxfun(@times, net.stim_subunits(fit_subs(1)).apply_NL_deriv( ...
                                  gint), stim_weights); 
            temp2 = fs.Xstims{1}' * sum(bsxfun(@times, temp1, residual), 3);        
            stim_grad([stim_indxs{:}]) = temp2(:);
        else
            for ii = 1:num_fit_subs 
                deriv = residual .* ...
                        (net.stim_subunits(fit_subs(ii)).apply_NL_deriv(gint(:,ii)) * ...
                        stim_weights(ii,:));
                stim_grad(stim_indxs{ii}) = sum( ...
                    (fs.Xstims{x_targets(ii)})' * deriv * mod_signs(ii), 2);
            end
        end
    else
        stim_grad = [];
        stim_weights_grad = [];
    end

    if net.fit_params.fit_overall_offsets
        % calculate derivatives with respect to constant terms (theta)
        offset_grad = sum(net.apply_spk_NL_deriv(G).*cost_grad, 1)';
    else
        offset_grad = [];
    end

    
    % ******************* COMPUTE REG VALUES AND GRADIENTS ****************
    if net.fit_params.fit_auto
        auto_reg_pen = 0.5 * lambda_w1 * sum(sum(w1.^2)) ...
                     + 0.5 * lambda_w2 * sum(sum(w2.^2)) ...
                     + 0.5 * lambda_b1 * sum(b1.^2)  ...        
                     + 0.5 * lambda_b2 * sum(b2.^2)  ...
                     + lambda_sp * sparse_func;
        if net.auto_subunit.weight_tie
            auto_reg_pen_grad = [2 * lambda_w2 * w2(:); ...
                                 lambda_b1 * b1; ...
                                 lambda_b2 * b2];
        else
            auto_reg_pen_grad = [lambda_w1 * w1(:); ...
                                 lambda_w2 * w2(:); ...
                                 lambda_b1 * b1; ...
                                 lambda_b2 * b2];
        end
        auto_grad = auto_grad / Z + auto_reg_pen_grad;
    else
        auto_reg_pen = 0;
    end

    if net.fit_params.fit_stim_individual || net.fit_params.fit_stim_shared
        stim_reg_pen = zeros(num_fit_subs,1);
        stim_reg_pen_grad = zeros(num_stim_indxs,1);
        for ii = 1:num_fit_subs
            stim_reg_pen(ii) = net.stim_subunits(fit_subs(ii)).reg_lambdas.l2;
            stim_reg_pen_grad(stim_indxs{ii}) = stim_reg_pen(ii)*filts{ii}(:);
        end
        stim_reg_pen = 0.5*stim_reg_pen'*cellfun(@(x) sum(sum(x.^2)), filts);
        stim_grad = stim_grad / Z + stim_reg_pen_grad;
    else
        stim_reg_pen = 0;
    end
    
    if net.fit_params.fit_stim_shared
        stim_weights_reg_pen = 0.5 * net.lambda_stim * sum(sum(stim_weights.^2));
        stim_weights_reg_pen_grad = net.lambda_stim * stim_weights(:);
        stim_weights_grad = stim_weights_grad(:) / Z + stim_weights_reg_pen_grad;
    else
        stim_weights_reg_pen = 0;
    end
    
    if net.fit_params.fit_overall_offsets
        offset_reg_pen = 0.5 * lambda_off * sum(offsets.^2);
        offset_reg_pen_grad = lambda_off * offsets;
        offset_grad = offset_grad / Z + offset_reg_pen_grad;
    else
        offset_reg_pen = 0;
    end
    
    
    % ******************* COMBINE *****************************************
    
    func = cost_func / Z + auto_reg_pen ...
                         + stim_reg_pen ...
                         + stim_weights_reg_pen ...
                         + offset_reg_pen;
                     
    grad = [auto_grad; stim_grad; stim_weights_grad; offset_grad];
    
    end % internal function

end % fit_weights method




