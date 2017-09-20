% This script shows how to use the rlvm to infer autoencoder latent 
% variables from simulated neural population activity.

%% load data

% create simulated data that only contains activity due to shared inputs
% (latent variables), i.e. no stimulus responses
data_struct = createSimData(0, 1);

% fit normalized 2p data
data = data_struct.data_2p';
data = bsxfun(@rdivide, data, std(data, [], 2));

%% fit coupling weights and latent vars using the autoencoder (2-photon)

% model: yhat = F[w2 * g(w1 * y + b1) + b2]
% - g() relu (non-negative latent variables)
% - F() linear

% construct initial model
num_cells = size(data, 1);
num_lvs = data_struct.lvs.num_lvs;
layers = [num_cells, num_lvs, num_cells]; % single-layer autoencoder
act_funcs = {'relu', 'lin'};    % g() is relu, F() is linear
net = RLVM( ...
    layers, ...                 % layer sizes, including input and output
    0, ...                      % zero stimulus subunits
    'noise_dist', 'gauss', ...  % gaussian noise dist for 2p data
    'act_funcs', act_funcs);

% set regularization parameters
net = net.set_reg_params( ...
    'layer', ...                % specify reg params for layers
    'l2_weights', 1e-4, ...     % l2 penalty on coupling weights
    'l2_biases', 1e-4);         % l2 penalty on biases

% set different optimization parameters
net = net.set_optim_params('display', 'iter');

% fit model
net = net.fit_model( ...
    'weights', ...              % specify fitting coupling weights
    'pop_activity', data);      % num_cells x T matrix of pop activity

%% display coupling weights
figure; 
subplot(121)
myimagesc(data_struct.lvs.coupling_mat);
title('True')
subplot(122)
myimagesc(net.layers(2).weights); % columns may be out of order
title('Estimated')
