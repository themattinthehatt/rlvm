function data_struct = createSimData(sim_stim, sim_lvs)
% Creates synthetic data to illustrate use of the rlvm. Intended to
% simulate 2-photon experiments in visual cortex, though generates the
% associated spiking data as well.
%
% INPUTS:
%   sim_stim    Boolean specifying whether or not stimulus responses to
%               oriented visual stimuli are simulated; each orientation is
%               presented during "stimulus" frames, then a number of blank
%               frames are presented before the next orientation. This
%               presentation is repeated for a predefined number of 
%               stimulus orientations, then the whole set of stimuli is
%               repeated for a predefined number of repetitions.
%   sim_lvs     Boolean specifying whether or not spontaneous activity is
%               simulated, using a predefined number of (possibly)
%               correlated, non-negative latent variables.
%
% OUTPUTS: 
%   data_struct structure that contains information about the simlulated
%               data

% overall constants
num_neurons = 50;
rng_seed = 0;
rng(rng_seed);

% latent variables constants
num_lvs = 5;
are_corr_lvs = 1;           % 1 for correlated latent variables
x = -5:1:5;
gauss_filt = exp(-x.^2/10); % gaussian filter for smoothing latent vars

% stim response constants
num_dirs = 12;
num_reps = 10;
num_blank_frames = 80;
num_stim_frames = 80;
if sim_stim
    T = num_dirs * num_reps * (num_blank_frames + num_stim_frames);
else
    T = 18000;              % pick an arbitrary experiment length
end

% 2p constants
dt   = 0.0625;              % time step size (16 Hz)
tau1 = 1.5;                 % decay time constant
gam1 = 1-dt/tau1;           % x1(t) = gam1*x1(t-1)
tau2 = .4;                  % rise time constant
gam2 = 1-dt/tau2;           % x2(t) = gam2*x2(t-1)
snr  = 5;                   % signal to noise ratio

% store firing rates for population
population_rates = zeros(T, num_neurons);

%% ********************** stimulus response activity **********************
% Neurons will have a gaussian-shaped tuning curve centered on their
% preferred frequency, which itself will be a uniformly distributed random
% variable. The max heights of the tuning curves will also be uniformly
% distributed between 0 and 1, and the temporal response will be a narrow
% gaussian.

% uniform random variable ranges
range_tuning_curve_widths  = [100, 10000];
range_tuning_curve_centers = [0, 359];
range_tuning_curve_heights = [0.1, 1];

% temporal response profile
x_temp = (-1:1:5)';
temporal_resp = exp(-x_temp.^2/10);

% get population parameters
m = range_tuning_curve_widths(1);
M = range_tuning_curve_widths(2);
curve_widths = m + (M - m) * rand(1, num_neurons);

m = range_tuning_curve_centers(1);
M = range_tuning_curve_centers(2);
curve_centers = m + (M - m) * rand(1, num_neurons);

m = range_tuning_curve_heights(1);
M = range_tuning_curve_heights(2);
curve_heights = m + (M - m) * rand(1, num_neurons);

% save stim information for rebuilding stim matrix later
blank_frames = cell(num_reps, num_dirs);
stim_frames = cell(num_reps, num_dirs);
epoch_frames = cell(num_reps, num_dirs);

if sim_stim
    
    % get tuning curves to wrap
    dirs1 = linspace(-360, -360/num_dirs, num_dirs);
    dirs2 = linspace(0, 360 - 360/num_dirs, num_dirs);
    dirs3 = linspace(360, 720 - 360/num_dirs, num_dirs);
    for rep = 1:num_reps
        for dir = 1:num_dirs

            % stimulus onset time
            t = (rep - 1) * (num_stim_frames + num_blank_frames) * num_dirs ...
              + (dir - 1) * (num_stim_frames + num_blank_frames) + 1;

            % record for stim data struct
            t_s = t + num_blank_frames;
            blank_frames{rep, dir} = t:(t_s-1);
            stim_frames{rep, dir} = t_s:(t_s+num_stim_frames-1);
            epoch_frames{rep, dir} = t:(t+num_blank_frames+num_stim_frames-1);

            % record population responses
            t1 = (dirs1(dir) - curve_centers).^2;
            t2 = (dirs2(dir) - curve_centers).^2;
            t3 = (dirs3(dir) - curve_centers).^2;
            exp_arg = min(t1, min(t2, t3));
            heights = curve_heights .* exp(-exp_arg./curve_widths);
            population_rates(t_s:t_s+length(x_temp)-1, :) = ...
                                                    temporal_resp * heights;

        end
    end
    
    % bring population rates up so something reasonable pops out of the
    % poisson random number generator
    population_rates = 6 * population_rates;

end


%% ********************** latent variable activity ************************

if sim_lvs
    
    % couplings between latent variables and neurons
    coupling_mat = createCouplingMat(num_neurons, num_lvs);

    % create correlated rvs
    if are_corr_lvs
        cov_mat = createLVCovarianceMat(num_lvs);
        [V,D] = eig(cov_mat);
        X = ((V*sqrt(D)) * randn(num_lvs,T))';
    else
        X = randn(num_lvs,T)';
    end

    % smooth noise
    X = conv2(X,gauss_filt');
    X = X((length(x)-1):(length(x)-2+T),:);

    % threshold to create non-negative latent variables
    thresh = 3;
    X = abs(X);
    X(X < thresh) = thresh;
    X = X - thresh;

    % create latent states that are convolved w/ calcium kernel
    Xsmooth = zeros(T,num_lvs);
    for n = 1:num_lvs
        % decay signal
        x1 = filter(1,[1 -gam1],X(:,n));
        % rise signal
        x2 = filter(1,[1 -gam2],X(:,n));
        % combine the two
        Xsmooth(:,n) = x1-x2;
    end

    % create firing rates for population
    population_rates = population_rates + X*coupling_mat';

else
    
    num_lvs = 0;
    coupling_mat = [];
    X = [];
    Xsmooth = [];
    
end

%% ********************** create data *************************************

% create spiking data
data_spikes = poissrnd(population_rates);

% create 2p data
calcium_sig = zeros(T,num_neurons);
for n = 1:num_neurons
    % decay signal
    x1 = filter(1, [1 -gam1], data_spikes(:,n));       
    % rise signal
    x2 = filter(1, [1 -gam2], data_spikes(:,n));       
    % combine the two
    calcium_sig(:,n) = x1 - x2;
end

% add noise with specified snr
sig_var = var(calcium_sig);
sig_var(sig_var == 0) = 1; % snr = 1 for empty channels if any
data_2p = calcium_sig + bsxfun(@times, ...
                               randn(T,num_neurons), ...
                               sqrt(sig_var/snr));

%% ********************** organize data ***********************************

data_struct.meta.num_neurons = num_neurons;
data_struct.meta.sim_lvs = sim_lvs;
data_struct.meta.sim_stim = sim_stim;
data_struct.meta.rng_seed = rng_seed;

data_struct.stim.num_dirs = num_dirs;
data_struct.stim.num_reps = num_reps;
data_struct.stim.num_blank_frames = num_blank_frames;
data_struct.stim.num_stim_frames = num_stim_frames;
data_struct.stim.blank_frames = blank_frames;
data_struct.stim.stim_frames = stim_frames;
data_struct.stim.epoch_frames = epoch_frames;
data_struct.stim.curve_widths = curve_widths;
data_struct.stim.curve_centers = curve_centers;
data_struct.stim.curve_heights = curve_heights;

data_struct.lvs.num_lvs = num_lvs;
data_struct.lvs.coupling_mat = coupling_mat;
data_struct.lvs.latent_vars = X;
data_struct.lvs.latent_vars_conv = Xsmooth;
data_struct.lvs.are_corr = are_corr_lvs;

data_struct.data_spikes = data_spikes;
data_struct.data_2p = data_2p;


