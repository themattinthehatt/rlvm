function [passes, fails, msg] = test_auto_stim_sh_models( ...
                                    data, noise_models, ...
                                    auto_NLtypes, stim_NLtypes, ...
                                    auto_regs_1, auto_regs_2, stim_regs)
   
% data should be a cell array with the same number of cells as there are
% noise models

passes = 0;
fails = 0;
msg = {};

fit_overall_offsets = [0, 1];
weight_tie = [0, 1];

% stim info - make two temporal subunits
stim1 = zeros(size(data{1}, 1), 1);
stim2 = stim1;
stim1(1:100:size(data{1}, 1)) = 1;
stim2(5:50:size(data{1}, 1)) = 1;
stim_params(1) = StimSubunit.create_stim_params([5, 1, 1], 1);
stim_params(2) = StimSubunit.create_stim_params([8, 1, 1], 1);
Xstims{1} = StimSubunit.create_time_embedding(stim1, stim_params(1));
Xstims{2} = StimSubunit.create_time_embedding(stim2, stim_params(2));

init_params = RLVM.create_init_params(stim_params, size(data{1}, 2), 5);

for i = 1:length(noise_models)
    for j = 1:length(auto_NLtypes)
        
        % models with regularization on parameters
        if ~isempty(auto_regs_1)
            for m = 1:length(fit_overall_offsets)
                for n = 1:length(weight_tie)
                    net = RLVM(init_params, ...
                               'noise_dist', noise_models{i}, ...
                               'act_func_hid', auto_NLtypes{j}, ...
                               'NL_types', stim_NLtypes{1}, ...
                               'fit_overall_offsets', fit_overall_offsets(m), ...
                               'weight_tie', weight_tie(n));
                    if strcmp(noise_models{i}, 'gauss')
                        net = net.set_params('spk_NL', 'lin');
                    elseif strcmp(noise_models{i}, 'poiss')
                        net = net.set_params('spk_NL', 'softplus');
                    end
                    for k = 1:length(auto_regs_1)
                        net = net.set_reg_params('auto', auto_regs_1{k}, 1);
                    end
                    for k = 1:length(stim_regs)
                        net = net.set_reg_params('stim', stim_regs{k}, 1);
                    end
                    net = net.set_fit_params('deriv_check', 1);
                    try
                        net = net.fit_model('params', data{i}, Xstims);
                        passes = passes + 1;
                    catch
                        fails = fails + 1;
                        msg{end+1} = sprintf('Failed on %s noise, %s act funcs, offset = %g, weight-tie = %g, param regs\n', ...
                                noise_models{i}, auto_NLtypes{j}, fit_overall_offsets(m), weight_tie(n));
                    end
                end
            end
        end
        
        % models with regulariztion on latent variables
        if ~isempty(auto_regs_2)
            for m = 1:length(fit_overall_offsets)
                for n = 1:length(weight_tie)
                    net = RLVM(init_params, ...
                               'noise_dist', noise_models{i}, ...
                               'act_func_hid', auto_NLtypes{j}, ...
                               'NL_types', stim_NLtypes{1}, ...
                               'fit_overall_offsets', fit_overall_offsets(m), ...
                               'weight_tie', weight_tie(n));
                    if strcmp(noise_models{i}, 'gauss')
                        net = net.set_params('spk_NL', 'lin');
                    elseif strcmp(noise_models{i}, 'poiss')
                        net = net.set_params('spk_NL', 'softplus');
                    end
                    for k = 1:length(auto_regs_1)
                        net = net.set_reg_params('auto', auto_regs_1{k}, 1);
                    end
                    for k = 1:length(stim_regs)
                        net = net.set_reg_params('stim', stim_regs{k}, 1);
                    end
                    net = net.set_fit_params('deriv_check', 1);
                    try
                        net = net.fit_model('params', data{i}, Xstims);
                        passes = passes + 1;
                    catch
                        fails = fails + 1;
                        msg{end+1} = sprintf('Failed on %s noise, %s act funcs, offset = %g, weight-tie = %g, latent var regs\n', ...
                                noise_models{i}, auto_NLtypes{j}, fit_overall_offsets(m), weight_tie(n));
                    end
                end
            end
        end
        
    end % act funcs
end % noise dist



