% This script shows how to use the rlvm to infer latent variables from
% simulated neural population activity, as well as simultaneously fit a
% simple psth-style stimulus model for each individual neuron. Just the 
% initial augmented-autoencoder network is fit here, though in principal
% the latent variables could be refit with the smoothing prior.
%
% The stimulus modeling capabilities of the rlvm mostly mirror the
% functionality of the Nonlinear Input Model (NIM); see
%
%   McFarland JM, Cui Y, Butts DA (2013) Inferring nonlinear neuronal 
%   computation based on physiologically plausible inputs. PLoS 
%   Computational Biology 9(7):e1003142
%
% and 
%
%   https://github.com/dbutts/NIMclass 
%
% for more information.

%% load data

% create simulated data that contains activity due to both shared inputs
% (latent variables) and stimulus responses
data_struct = createSimData(1, 1);

% % fit normalized 2p data
data = data_struct.data_2p;
data = bsxfun(@rdivide, data, std(data));

%% create stimulus matrix

% specify stimulus parameters
% use 20 lags at 8 bin intervals to cover the full 160-bin duration of stim
% presentation + blank presentation. This effectively represents the psth
% at a lower temporal resolution, which implicitly regularizes it
num_lags = 20;
tent_spacing = 8;
stim_params = StimSubunit.create_stim_params( ...
            [num_lags, data_struct.stim.num_dirs, 1], ... % stim dimensions
            size(data, 2), ...                            % number of stim models
            'tent_spacing', tent_spacing);                % bins per psth parameter
       
% fit lagged stim model regressing on visual stimulus onset
stim_frames = data_struct.stim.stim_frames;
vis_stim_onsets = zeros(size(data, 1), data_struct.stim.num_dirs);
for i = 1:data_struct.stim.num_reps
    for j = 1:data_struct.stim.num_dirs
        vis_stim_onsets(stim_frames{i,j}(1),j) = 1;
    end
end

% create lagged stimulus matrix
Xstims{1} = StimSubunit.create_time_embedding(vis_stim_onsets, stim_params);
% scale entries so that stimulus model parameters are roughly the same 
% magnitude as coupling matrix weights
Xstims{1} = bsxfun(@rdivide, Xstims{1}, std(Xstims{1})); 

%% fit coupling weights and latent vars using the autoencoder (2-photon)

% model: yhat = F[w2 * g(w1 * y + b1) + b2 + f(k * s)]
%   latent variable components
%       - g() relu (non-negative latent variables)
%   stimulus components
%       - f() linear
%       - k a linear filter on the stimulus
%   F() linear

% construct initial model        
layers = [size(data,2), data_struct.lvs.num_lvs, size(data,2)];
act_funcs = {'relu', 'lin'};
net = RLVM(layers, 1, ...
    'stim_params', stim_params, ...
    'noise_dist', 'gauss', ...  % gaussian noise dist for 2p data
    'act_funcs', act_funcs, ... % g() is relu, F() is linear
    'NL_types', 'lin');         % f() is linear

% set latent variable regularization parameters
net = net.set_reg_params('layer', ...    
    'l2_weights', 1e-4, ...     % l2 penalty on coupling weights
    'l2_biases', 1e-5);         % l2 penalty on biases

% set stimulus model regularization parameters
net = net.set_reg_params('stim', ...    
    'l2', 1e-6);                % l2 penalty on psth values
            
% 'params' specifies that we are fitting model params 
% (coupling weights and stim models; warning - may take some time)
net = net.set_optim_params( ...
    'display', 'iter', ...
    'max_iter', 5000, ...
    'deriv_check', 1);
tic
net = net.fit_model( ...
    'weights', ...
    'pop_activity', data', ...
    'Xstims', Xstims, ...
    'indx_tr', 1:1000);
t = toc

%% display model components 

% coupling weights
figure; 
subplot(321)
myimagesc(data_struct.lvs.coupling_mat);
title('True')
subplot(322)
myimagesc(net.layers(2).weights);        % columns may be out of order
title('Estimated')

% example neuron psths
subplot(323)
myimagesc(reshape(net.stim_subunits.filt(:,20), num_lags, []));
% xlabel('Stim #')
ylabel('Time lags (bins)')

subplot(324)
myimagesc(reshape(net.stim_subunits.filt(:,40), num_lags, []));
% xlabel('Stim #')
% ylabel('Time lags (bins)')

subplot(325)
myimagesc(reshape(net.stim_subunits.filt(:,60), num_lags, []));
xlabel('Stim #')
ylabel('Time lags (bins)')

subplot(326)
myimagesc(reshape(net.stim_subunits.filt(:,80), num_lags, []));
xlabel('Stim #')
% ylabel('Time lags (bins)')

