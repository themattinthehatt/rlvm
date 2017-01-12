% This script shows how to use the rlvm to infer latent variables from
% simulated neural population activity. First an autoencoder is fit to the
% data to produce the initial estimates of the latent variables and the
% coupling matrix from latent variables to individual neurons. Then, given
% the coupling matrix, the latent variables are fit using a smoothing
% prior.

%% load data

% create simulated data that only contains activity due to shared inputs
% (latent variables), i.e. no stimulus responses
data_struct = createSimData(0, 1);

% fit normalized 2p data
data = data_struct.data_spikes;

%% fit coupling weights and latent vars using the autoencoder (2-photon)

% model: yhat = F[w2 * g(w1 * y + b1) + b2]
% - g() relu (non-negative latent variables)
% - F() log(1+exp())
% - w1 = w2' (weight-tying)

% initialize model parameters
init_params = RLVM.create_init_params( ...
            [], ...                     % no stimulus model parameters
            size(data, 2), ...          % number of neurons
            data_struct.lvs.num_lvs);   % number of latent vars to fit

% construct initial model
net = RLVM(init_params, ...
            'noise_dist', 'poiss', ...  % gaussian noise dist for 2p data
            'act_func_hid', 'relu', ... % g() is relu
            'weight_tie', 0, ...        % weight-tie constraint
            'spk_NL', 'softplus');           % F() is linear

% set regularization parameters
net.auto_subunit = net.auto_subunit.set_reg_params( ...
           'l2_weights1', 1e-2, ...     % l2 penalty on coupling weights
           'l2_biases1', 1, ...         % l2 penalty on biases
           'l2_weights2', 1e-2, ...
           'l2_biases2', 0);
                   
% 'params' specifies that we are fitting model params (coupling weights)
net = net.set_optim_params('display', 'iter');
net = net.fit_model('params', data);

%% display coupling weights

figure; 
subplot(121)
myimagesc(data_struct.lvs.coupling_mat);
title('True')
subplot(122)
myimagesc(net.auto_subunit.w2');        % columns may be out of order
title('Estimated')

%% smooth latent vars, initialized with autoencoder values (2-photon)

% get "autoencoder" latent variables
[~, ~, mod_internals] = net.get_model_eval(data);
latent_vars_pre = mod_internals.auto_fgint{1};

% fit smoothed latent variables
net = net.set_reg_params('auto', 'd2t_hid', 1); % set smoothing reg param
net = net.fit_model('latent_vars', data, ...            
            'init_weights', 'model', ...        % use auto weights to init
            'init_latent_vars', 'model');       % use auto lvs to init

% get smoothed latent variables
[~, ~, mod_internals] = net.get_model_eval(data);
latent_vars_post = mod_internals.auto_fgint{1};

% plot examples
figure;
ax(1) = subplot(211);
plot(latent_vars_pre(:,1));
xlim([1, 500])
title('Pre-smoothing')
ax(2) = subplot(212);
plot(latent_vars_post(:,1));
xlim([1, 500])
title('Post-smoothing')
linkaxes(ax, 'xy')

