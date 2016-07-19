function [net, weights, latent_vars] = fit_alt(net, fs)
% net = net.fit_alt(fitting_struct)
%
% alternates between fitting latent states and model parameters
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

% ************************** DEFINE DEFAULTS ******************************
num_iters = 10;
verbose = 0;
if verbose
    fprintf('Beginning alternating fitting routine\n')
end

% ************************** CHECK INPUTS *********************************

% ensure size of pop_activity is consistent with autoencoder weights
net.auto_subunit.check_inputs(fs.pop_activity);

% ************************** SET INITIAL VALUES ***************************
T = size(fs.pop_activity, 1);
num_cells = net.num_cells;
num_hid_nodes = net.auto_subunit.num_hid_nodes;

% set initial values here and save in RLVM object
if strcmp(fs.init_type, 'auto')
    if verbose
        fprintf('Initializing with autoencoder...')
    end
    net = net.fit_weights_latent_vars(fs);
    if verbose
        fprintf('Done!\n')
    end
    % make sure these values are subsequently used
    fs.init_latent_vars = 'model';
    fs.init_weights = 'model';
elseif strcmp(fs.init_type, 'init')
    % latent vars
    if ischar(fs.init_latent_vars)
        if strcmp(fs.init_latent_vars, 'gauss')
            net.auto_subunit.latent_vars = abs(randn(T, num_hid_nodes));
            fs.init_latent_vars = 'model';
        else
            fs.init_latent_vars = 'model';
        end
    elseif ismatrix(fs.init_latent_vars)
        assert(size(fs.init_latent_vars) == [T, num_hid_nodes], ...
            'init_latent_vars matrix does not have proper size')
        net.auto_subunit.latent_vars = fs.init_latent_vars;
        fs.init_latent_vars = 'model';
    else
        error('Improper init_latent_vars format')
    end
    % weights
    if ischar(fs.init_weights)
        if strcmp(fs.init_weights, 'gauss')
            net.auto_subunit.w2 = 0.1 * randn(size(net.auto_subunit.w2));
            net.auto_subunit.b2 = 0.1 * randn(size(net.auto_subunit.b2));
            fs.init_weights = 'model';
        else
            fs.init_weights = 'model';
        end
    elseif isvector(fs.init_weights)
        assert(length(fs.init_weights) == ...
            num_cells + net.num_cells * num_hid_nodes, ...
            'init_weight vector does not have proper length')
        [net.auto_subunit.w2, net.auto_subunit.b2] = ...
                    net.auto_subunit.get_decoding_weights(fs.init_weights);

    else
        error('Improper init_weight format')
    end
end

% ************************** FIT MODEL ************************************

% default is to fit latent vars first
if strcmp(fs.first_fit, 'weights')
    if verbose
        fprintf('Iter 0:\n')
        fprintf('\tFitting weights\n')
    end
    net = net.fit_weights(fs);
end

iter = 1;
while iter <= num_iters
    
    if verbose
        fprintf('\nIter %g:\n', iter)
    end
    
    % fit latent states
    if verbose
        fprintf('\tFitting latent states...\n')
    end
    net = net.fit_latent_vars(fs);
    
    % fit weights
    if verbose
        fprintf('\tFitting weights...\n')
    end
    net = net.fit_weights(fs);
    
    % print updates
    if verbose
        [~, cost_func] = net.get_model_eval(fs.pop_activity, fs.Xstims);
        fprintf('\tCost function = %g', cost_func)
    end
    
    iter = iter + 1;
    
end
fprintf('\n')

% output
weights = net.auto_subunit.w2';
latent_vars = net.auto_subunit.latent_vars;