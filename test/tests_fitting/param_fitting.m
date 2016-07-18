%% load data
load('~/Dropbox/Lab/auto_paper/sim_data/data/sim_data_21.mat')
data = data(1:3000, 1:10);
data_spike = data_spike(1:3000, 1:10);

%% auto model

init_params = RLVM.create_init_params([], size(data, 2), 5);

net1 = RLVM(init_params, 'weight_tie', 1);
net1 = net1.set_reg_params('auto', 'l2_biases1', 1e-5, 'l2_biases2', 1e-5, 'l2_weights', 1e-4);
net1 = net1.fit_model('params', data);
[r21, cost_val1, mod_int1, reg_params1] = net1.get_model_eval(data);

%% stim_ind model

stim1 = zeros(size(data, 1), 1);
stim2 = stim1;
stim1(1:100:size(data, 1)) = 1;
stim2(5:50:size(data, 1)) = 1;
stim_params(1) = StimSubunit.create_stim_params([5, 1, 1], size(data, 2));
stim_params(2) = StimSubunit.create_stim_params([8, 1, 1], size(data, 2));
Xstims{1} = StimSubunit.create_time_embedding(stim1, stim_params(1));
Xstims{2} = StimSubunit.create_time_embedding(stim2, stim_params(2));

init_params = RLVM.create_init_params(stim_params, size(data, 2), 0);

net2 = RLVM(init_params, 'noise_dist', 'poiss', 'NL_types', 'lin');
net2 = net2.set_reg_params('stim', 'l2', 1);
net2 = net2.fit_model('params', data_spike, Xstims);
% [r22, cost_val2, mod_int2, reg_params2] = net2.get_model_eval(data, Xstims);

%% auto + stim_ind model
% 
% stim = zeros(size(data, 1), 1);
% stim(1:100:size(data, 1)) = 1;
% stim_params = StimSubunit.create_stim_params([5, 1, 1], size(data, 2));
% Xstims{1} = StimSubunit.create_time_embedding(stim, stim_params);
% init_params = RLVM.create_init_params(stim_params, size(data, 2), 5);
% 
% net3 = RLVM(init_params);
% net3 = net3.set_reg_params('auto', 'l2_biases', 100, 'l2_weights', 1000);
% net3 = net3.set_reg_params('stim', 'l2', 1);
% net3 = net3.fit_model(data, Xstims);
% % [r23, cost_val3, mod_int3, reg_params3] = net3.get_model_eval(data, Xstims);