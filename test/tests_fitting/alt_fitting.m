%% load data
load('~/Dropbox/Lab/auto_paper/sim_data/data/sim_data_21.mat')
data = data(1:5000, :);

num_iters = 3; % number of iterations for alternating fit routines

% Notes
% seems it will be best to store the weights learned from the alternating
% fitting routine in the auto_subunit under w2 and b2; maybe once this
% happens it is best to get rid of w1 and b1? good at least for throwing
% errors. The latent states can then also be stored with the model. In
% evaluating the model (AutoSubunit.get_model_internals), method can look
% to see if w1/b1 are gone & ~isempty(latent_vars), and use this to produce
% gint and fgint

%% auto model

% store model fitting results
r21 = cell(num_iters, 2);
cost_val1 = cell(num_iters, 2);
mod_int1 = cell(num_iters, 2);
reg_params1 = cell(num_iters, 2);
weights1 = cell(num_iters, 2);
latent_vars1 = cell(num_iters, 2);

% initialize model
init_params = RLVM.create_init_params([], size(data, 2), 5);
net1 = RLVM(init_params, 'weight_tie', 1, 'act_func_hid', 'relu');

% initial fit
net1 = net1.set_reg_params('auto', ...
                           'l2_biases1', 1e-5, ...
                           'l2_biases2', 1e-5, ...
                           'l2_weights', 1e-4);
net1 = net1.fit_model('params', data);

[r2_0, cost_val_0, mod_int_0, reg_params_0] = ...
    net1.get_model_eval(data);

%% plotting true vs. learned weights
figure;
subplot(121)
myimagesc(spont_clusts);
subplot(122)
w2 = net1.auto_subunit.w2';
w2 = bsxfun(@times, w2, [1, -1, -1, 1, 1]);
w2 = w2(:, [5, 4, 3, 1, 2]);
myimagesc(w2)

%% smoothed latent states test
net1 = net1.set_reg_params('auto', 'd2t_hid', 1e-2);
net1 = net1.set_fit_params('deriv_check', 0);
[net2, ~, latent_vars] = net1.fit_model('latent_vars', data(1:5000, :));

%% plotting smoothed latent states
figure; 
ax(1) = subplot(311);
plot(mod_int_0.auto_fgint{1});
ax(2) = subplot(312);
latent_vars(latent_vars < 0) = 0;
plot(latent_vars)
ax(3) = subplot(313);
plot(Xsmooth(1:5000, :));
linkaxes(ax, 'x')

%% fitting weights test
net2 = net2.set_fit_params('deriv_check', 0);
[net3, ~, latent_vars] = net2.fit_model('weights', data(1:5000, :));

%% plotting smoothed latent states
figure;
subplot(131)
myimagesc(spont_clusts);
subplot(132)
w2 = net2.auto_subunit.w2';
w2 = w2(:, [4, 5, 3, 2, 1]);
myimagesc(w2)
subplot(133)
w2_ = net3.auto_subunit.w2';
w2_ = w2_(:, [4, 5, 3, 2, 1]);
myimagesc(w2_)









