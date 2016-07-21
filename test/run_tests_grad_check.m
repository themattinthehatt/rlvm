% Script to run gradient checks on RLVM.
% When running all gradient checks, this script can take quite a while to
% run (on the order of tens of minutes)
%
% TODO
%   - base stim_sh
%   - base auto_stim_sh
%   - more complicated stim models (spatial components)

%% setup 

% output detailed results
verbose = 1;

% select tests to run
tests.auto = 1;
tests.stim_ind = 1;
tests.stim_sh = 1;
tests.auto_stim_ind = 1;
tests.auto_stim_sh = 1;

% define all model options
noise_models = {'gauss', 'poiss'};

auto_NLtypes = {'lin', 'relu', 'sigmoid', 'softplus'};
auto_regs_1 = {'l2_biases', 'l2_weights'};
auto_regs_2 = {'l1_hid'};

stim_NLtypes = {'lin', 'relu', 'softplus'};
stim_regs = {'l2'};

% import test data
% load('../data/sim_data.mat')
load('~/Dropbox/Lab/auto_paper/sim_data/data/sim_data_21.mat')
test_data{1} = data(1:3000, 1:10);
test_data{1} = data_spike(1:3000, 1:10);

%% run tests

pass_total = 0;
fail_total = 0;
for i = 1:length(noise_models)
    
    % auto model
    if tests.auto
        [passes, fails, auto_msg] = test_auto_models( ...
            test_data, ...
            noise_models, ...
            auto_NLtypes, ...
            auto_regs_1, ...
            auto_regs_2);
        pass_total = pass_total + passes;
        fail_total = fail_total + fails;
    end

    % stim_individual model
    if tests.stim_ind
        [passes, fails, stim_ind_msg] = test_stim_ind_models( ...
            test_data, ...
            noise_models, ...
            stim_NLtypes, ...
            stim_regs);
        pass_total = pass_total + passes;
        fail_total = fail_total + fails;
    end
    
    % stim_shared model
    if tests.stim_sh
        [passes, fails, stim_sh_msg] = test_stim_sh_models( ...
            test_data, ...
            noise_models, ...
            stim_NLtypes, ...
            stim_regs);
        pass_total = pass_total + passes;
        fail_total = fail_total + fails;
    end
    
    % auto+stim_individual model
    if tests.auto_stim_ind
        [passes, fails, auto_stim_ind_msg] = test_auto_stim_ind_models( ...
            test_data, ...
            noise_models, ...
            auto_NLtypes, ...
            stim_NLtypes, ...
            auto_regs_1, ...
            auto_regs_2, ...
            stim_regs);
        pass_total = pass_total + passes;
        fail_total = fail_total + fails;
    end
    
    % auto+stim_shared model
    if tests.auto_stim_sh
        [passes, fails, auto_stim_sh_msg] = test_auto_stim_sh_models( ...
            test_data, ...
            noise_models, ...
            auto_NLtypes, ...
            stim_NLtypes, ...
            auto_regs_1, ...
            auto_regs_2, ...
            stim_regs);
        pass_total = pass_total + passes;
        fail_total = fail_total + fails;
    end
    
end

%% output results

fprintf('============================================================\n')
fprintf('                            RESULTS                         \n')
fprintf('============================================================\n\n')

if tests.auto
    fprintf('AUTO MODEL\n')
    if ~isempty(auto_msg)
        for i = 1:length(auto_msg)
            fprintf(auto_msg{i})
        end
    else
        fprintf('All tests passed\n')
    end
    fprintf('\n')
end

if tests.stim_ind
    fprintf('STIM IND MODEL\n')
    if ~isempty(stim_ind_msg)
        for i = 1:length(stim_ind_msg)
            fprintf(stim_ind_msg{i})
        end
    else
        fprintf('All tests passed\n')
    end
    fprintf('\n')
end

if tests.stim_sh
    fprintf('STIM SH MODEL\n')
    if ~isempty(stim_sh_msg)
        for i = 1:length(stim_sh_msg)
            fprintf(stim_sh_msg{i})
        end
    else
        fprintf('All tests passed\n')
    end
    fprintf('\n')
end

if tests.auto_stim_ind
    fprintf('AUTO STIM IND MODEL\n')
    if ~isempty(auto_stim_ind_msg)
        for i = 1:length(auto_stim_ind_msg)
            fprintf(auto_stim_ind_msg{i})
        end
    else
        fprintf('All tests passed\n')
    end
    fprintf('\n')
end

if tests.auto_stim_sh
    fprintf('AUTO STIM SH MODEL\n')
    if ~isempty(auto_stim_sh_msg)
        for i = 1:length(auto_stim_sh_msg)
            fprintf(auto_stim_sh_msg{i})
        end
    else
        fprintf('All tests passed\n')
    end
    fprintf('\n')
end

fprintf('\nGradient Check Totals\n')
fprintf('\tPasses: %g\n', pass_total)
fprintf('\tFails: %g\n', fail_total)