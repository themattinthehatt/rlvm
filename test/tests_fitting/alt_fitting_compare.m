%% load data
load('~/Dropbox/Lab/auto_paper/sim_data/data/sim_data_21.mat')
data = data(1:5000, :);

num_iters = 3; % number of iterations for alternating fit routines

%% alt fit w/ auto init

% store model fitting results
r2_auto = cell(num_iters, 2);
cost_val_auto = cell(num_iters, 2);
mod_int_auto = cell(num_iters, 2);
reg_params_auto = cell(num_iters, 2);
weights_auto = cell(num_iters, 2);
latent_vars_auto = cell(num_iters, 2);
net_auto(num_iters+1, 2) = RLVM();

% initialize model
init_params = RLVM.create_init_params([], size(data, 2), 5);
net_auto(1,2) = RLVM(init_params, 'weight_tie', 1, 'act_func_hid', 'relu');

% initial fit
fprintf('Fitting initial auto model\n')
net_auto(1,2) = net_auto(1,2).set_reg_params('auto', ...
                           'l2_biases1', 1e-5, ...
                           'l2_biases2', 1e-5, ...
                           'l2_weights', 1e-4, ...
                           'd2t_hid', 1e-2);
net_auto(1,2) = net_auto(1,2).fit_model('params', data);

[r2_0, cost_val_0, mod_int_0, reg_params_0] = ...
    net_auto(1,2).get_model_eval(data);
fprintf('initial r2: %g\n\n', mean(r2_0))

for i = 1:num_iters
    fprintf('EM iter %g:\n', i)
    
    % fit latent vars given weights
    [net_auto(i+1,1), weights_auto{i,1}, latent_vars_auto{i,1}] = ...
        net_auto(i,2).fit_model( ...
            'latent_vars', data, ...
            'init_weights', 'model', ...
            'init_latent_vars', 'model');
    [r2_auto{i,1}, cost_val_auto{i,1}, mod_int_auto{i,1}, reg_params_auto{i,1}] = ...
        net_auto(i+1,1).get_model_eval(data);
    fprintf('\tlatent_vars r2: %g\n', mean(r2_auto{i,1}))
    
    % fit weights given latent vars
    [net_auto(i+1,2), weights_auto{i,2}, latent_vars_auto{i,2}] = ...
        net_auto(i+1,1).fit_model( ...
            'weights', data, ...
            'init_weights', 'model', ...
            'init_latent_vars', 'model');
    [r2_auto{i,2}, cost_val_auto{i,2}, mod_int_auto{i,2}, reg_params_auto{i,2}] = ...
        net_auto(i+1,2).get_model_eval(data);
    fprintf('\tweights r2: %g\n', mean(r2_auto{i,2}))
    
end

new_order = [4, 5, 3, 2, 1];
% plot latent vars
figure;
for i = 1:num_iters
    ax(i) = subplot(num_iters+1, 1, i);
    latent_vars = latent_vars_auto{i,1}(1:500,:);
    latent_vars(latent_vars < 0) = 0;
    plot(latent_vars(:,new_order));
end
ax(num_iters+1) = subplot(num_iters+1, 1, num_iters+1);
plot(Xsmooth(1:500,:));
linkaxes(ax, 'x')

% plot weight matrices
figure;
for i = 1:num_iters
    subplot(1, num_iters+1, i)
    w2 = net_auto(i+1,2).auto_subunit.w2';
    w2 = w2(:,new_order);
    myimagesc(w2)
end
subplot(1, num_iters+1, num_iters+1)
myimagesc(spont_clusts)

%% alt fit w/ rand init

% store model fitting results
r2_rand = cell(num_iters, 2);
cost_val_rand = cell(num_iters, 2);
mod_int_rand = cell(num_iters, 2);
reg_params_rand = cell(num_iters, 2);
weights_rand = cell(num_iters, 2);
latent_vars_rand = cell(num_iters, 2);
net_rand(num_iters+1, 2) = RLVM();

% initialize model
init_params = RLVM.create_init_params([], size(data, 2), 5);
net_rand(1,2) = RLVM(init_params, 'act_func_hid', 'relu');

% no initial fit
fprintf('Random initial auto model\n')
net_rand(1,2) = net_rand(1,2).set_reg_params('auto', ...
                           'l2_biases1', 1e-5, ...
                           'l2_biases2', 1e-5, ...
                           'l2_weights', 1e-4, ...
                           'd2t_hid', 1e-2);
net_rand(1,2).auto_subunit.w2 = 0.2 * randn(5, size(data,2));
net_rand(1,2).auto_subunit.latent_vars = abs(randn(5000, 5));

[r2_0, cost_val_0, mod_int_0, reg_params_0] = ...
    net_rand(1,2).get_model_eval(data);
fprintf('initial r2: %g\n\n', mean(r2_0))

for i = 1:num_iters
    fprintf('EM iter %g:\n', i)
    
    % fit latent vars given weights
    [net_rand(i+1,1), weights_rand{i,1}, latent_vars_rand{i,1}] = ...
        net_rand(i,2).fit_model( ...
            'latent_vars', data, ...
            'init_weights', 'model', ...
            'init_latent_vars', 'model');
    [r2_rand{i,1}, cost_val_rand{i,1}, mod_int_rand{i,1}, reg_params_rand{i,1}] = ...
        net_rand(i+1,1).get_model_eval(data);
    fprintf('\tlatent_vars r2: %g\n', mean(r2_rand{i,1}))
    
    % fit weights given latent vars
    [net_rand(i+1,2), weights_rand{i,2}, latent_vars_rand{i,2}] = ...
        net_rand(i+1,1).fit_model( ...
            'weights', data, ...
            'init_weights', 'model', ...
            'init_latent_vars', 'model');
    [r2_rand{i,2}, cost_val_rand{i,2}, mod_int_rand{i,2}, reg_params_rand{i,2}] = ...
        net_rand(i+1,2).get_model_eval(data);
    fprintf('\tweights r2: %g\n', mean(r2_rand{i,2}))
    
end

new_order = [4, 5, 1, 2, 3];
% plot latent vars
figure;
for i = 1:num_iters
    ax(i) = subplot(num_iters+1, 1, i);
    latent_vars = latent_vars_rand{i,1}(1:500,:);
    latent_vars(latent_vars < 0) = 0;
    plot(latent_vars(:,new_order));
end
ax(num_iters+1) = subplot(num_iters+1, 1, num_iters+1);
plot(Xsmooth(1:500,:));
linkaxes(ax, 'x')

% plot weight matrices
figure;
for i = 1:num_iters
    subplot(1, num_iters+1, i)
    w2 = net_rand(i+1,2).auto_subunit.w2';
    w2 = w2(:,new_order);
    myimagesc(w2)
end
subplot(1, num_iters+1, num_iters+1)
myimagesc(spont_clusts)