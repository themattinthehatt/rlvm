classdef RLVMTests < matlab.unittest.TestCase

% unit testing for RLVM class methods
% TOFIX
% - set_optim_params
%   - batch_size
% - set_stim_params
%   - > 1 subunits; make sure both are updated by default, and that 
%     individual stimsubunits can be adjusted
% - get_model_eval
%   - more extensive
%   - check w/ stim subunits
% - fit_model?
%   - check key-value pair handling

properties 
    num_lags = 8;
    num_xpix = 4;
    num_ypix = 2;
    num_cells = 8;
    num_hid_nodes = 2;
    expt_len = 100;
    stim_params
    init_params
    net
    xstim
end

methods (TestMethodSetup)

    function problem_setup(test_case)
        test_case.stim_params = StimSubunit.create_stim_params( ...
            [test_case.num_lags, test_case.num_xpix, test_case.num_ypix], ...
            test_case.num_cells);
        test_case.xstim{1} = StimSubunit.create_time_embedding(...
            randn(test_case.expt_len, test_case.num_xpix*test_case.num_ypix), ...
            test_case.stim_params);
        test_case.init_params = RLVM.create_init_params( ...
            test_case.stim_params, test_case.num_cells, test_case.num_hid_nodes);
        test_case.net = RLVM(test_case.init_params);
        warning('off', 'all')
    end

end

methods (TestMethodTeardown)
    
    function problem_teardown(test_case)
        warning('on', 'all')
    end
    
end

methods (Test)

    function test_setup(test_case)
        % test that problem_setup executed correctly; kill test if this is
        % not the case
        test_case.fatalAssertEqual(test_case.stim_params.dims, [test_case.num_lags, test_case.num_xpix, test_case.num_ypix])
        test_case.fatalAssertEqual(test_case.init_params.stim_params.dims, test_case.stim_params.dims)
        test_case.fatalAssertEqual(test_case.init_params.num_subunits, 1)
        test_case.fatalAssertEqual(test_case.init_params.num_cells, test_case.num_cells)
        test_case.fatalAssertEqual(test_case.init_params.num_hid_nodes, test_case.num_hid_nodes)
        test_case.fatalAssertEqual(test_case.net.fit_params.fit_auto, 1)
        test_case.fatalAssertEqual(test_case.net.fit_params.fit_stim_individual, 1)
        test_case.fatalAssertEqual(test_case.net.fit_params.fit_stim_shared, 0)
        test_case.fatalAssertEqual(test_case.net.fit_params.fit_overall_offsets, 0)
    end
        
    function test_RLVM(test_case)
        % test constructor
        test_case.verifyNotEmpty(test_case.net.stim_subunits)
        test_case.verifyNotEmpty(test_case.net.auto_subunit)
        test_case.verifyClass(RLVM(), 'RLVM') 
    end

    function test_setting_methods(test_case)
        % set_params
        temp = test_case.net.set_params('noise_dist', 'poiss', ...
                                        'spk_NL', 'softplus', ...
                                        'num_cells', 30);
        test_case.verifySubstring(temp.noise_dist, 'poiss')
        test_case.verifySubstring(temp.spk_NL, 'softplus')
        test_case.verifyEqual(temp.num_cells, 30)
        test_case.verifyEqual(temp.auto_subunit.num_cells, 30)
        
        % set_fit_params
        temp = test_case.net.set_fit_params('fit_auto', 1);
        test_case.verifyEqual(temp.fit_params.fit_auto, 1)
        temp = test_case.net.set_fit_params('fit_auto', 0);
        test_case.verifyEqual(temp.fit_params.fit_auto, 0)
        temp = test_case.net.set_fit_params('fit_stim_individual', 0, ...
                                            'fit_stim_shared', 0);
        temp2 = temp.set_fit_params('fit_stim_individual', 1);
        test_case.verifyEqual(temp2.fit_params.fit_stim_individual, 1)
        temp2 = temp.set_fit_params('fit_stim_individual', 0);
        test_case.verifyEqual(temp2.fit_params.fit_stim_individual, 0)
        temp2 = temp.set_fit_params('fit_stim_shared', 1);
        test_case.verifyEqual(temp2.fit_params.fit_stim_shared, 1)
        temp2 = temp.set_fit_params('fit_stim_shared', 0);
        test_case.verifyEqual(temp2.fit_params.fit_stim_shared, 0)
        temp2 = temp.set_fit_params('fit_overall_offsets', 1);
        test_case.verifyEqual(temp2.fit_params.fit_overall_offsets, 1)
        temp2 = temp.set_fit_params('fit_overall_offsets', 0);
        test_case.verifyEqual(temp2.fit_params.fit_overall_offsets, 0) 
        test_case.verifyError(@() test_case.net.set_fit_params( ...
            'fit_stim_individual', 1, ...
            'fit_stim_shared', 1), ...
            'RLVM:invalidoption');
        test_case.verifyError(@() test_case.net.set_fit_params( ...
            'fit_stim_shared', 1, ...
            'fit_stim_individual', 1), ...
            'RLVM:invalidoption');
        temp = test_case.net.set_fit_params('deriv_check', 1);
        test_case.verifyEqual(temp.fit_params.deriv_check, 1)
        temp = test_case.net.set_fit_params('deriv_check', 0);
        test_case.verifyEqual(temp.fit_params.deriv_check, 0)
        
        % set_optim_params
        temp = test_case.net.set_optim_params('optimizer', 'fminunc', ...
                                              'display', 'batch', 'max_iter', 5, 'monitor', 'batch');
        test_case.verifySubstring(temp.optim_params.optimizer, 'fminunc')
%         test_case.verifyEqual(temp.optim_params.batch_size, 1000)
        test_case.verifySubstring(temp.optim_params.Display, 'batch')
        test_case.verifyEqual(temp.optim_params.maxIter, 5)
        test_case.verifySubstring(temp.optim_params.monitor, 'batch')
                
        % set_reg_params
        temp = test_case.net.set_reg_params('stim', 'l2', 1, 'd2t', 1, 'd2x', 1, 'd2xt', 1);
        test_case.verifyEqual(temp.stim_subunits.reg_lambdas.l2, 1)
        test_case.verifyEqual(temp.stim_subunits.reg_lambdas.d2t, 1)
        test_case.verifyEqual(temp.stim_subunits.reg_lambdas.d2x, 1)
        test_case.verifyEqual(temp.stim_subunits.reg_lambdas.d2xt, 1)
        
        temp = test_case.net.set_reg_params('auto', 'l2_biases', 1, 'l2_weights', 1, 'l1_hid', 1, 'd2t_hid', 1);
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_biases1, 1)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_biases2, 1)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_weights1, 1)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_weights2, 1)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l1_hid, 1)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.d2t_hid, 1)
        
        temp = test_case.net.set_reg_params('auto', 'l2_biases1', 10, 'l2_biases2', 10, ...
                                                    'l2_weights1', 10, 'l2_weights2', 10);
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_biases1, 10)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_biases2, 10)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_weights1, 10)
        test_case.verifyEqual(temp.auto_subunit.reg_lambdas.l2_weights2, 10)
    end

    function test_getting_methods(test_case)
        % get_model_eval
        data = randn(test_case.expt_len, test_case.net.num_cells);
        temp = test_case.net.set_fit_params('fit_stim_individual', 0);
        temp = temp.set_fit_params('fit_stim_shared', 0);
        [~, ~, mod_int, ~] = temp.get_model_eval(data, test_case.xstim, 'indx_tr', 1:10);
        test_case.verifySize(mod_int.auto_fgint{1}, [10, test_case.net.auto_subunit.num_hid_nodes])
    end

    function test_fitting_methods(test_case)
        % fit_model
        data = randn(test_case.expt_len, test_case.net.num_cells);
        temp = test_case.net.set_fit_params('fit_stim_individual', 0);
        temp = temp.set_fit_params('fit_stim_shared', 0);
        temp = temp.set_optim_params('max_iter', 2);
        temp = temp.fit_model('params', data, test_case.xstim, 'indx_tr', 1:10);
        test_case.verifyNotEqual(temp.auto_subunit.w2, test_case.net.auto_subunit.w2')
    end
    
    function test_hidden_methods(test_case)
        % apply_spk_NL/apply_spk_NL_deriv
        data = randn(100, test_case.num_cells);
        
        temp = test_case.net.set_params('spk_NL', 'lin');
        output = temp.apply_spk_NL(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_spk_NL_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = test_case.net.set_params('spk_NL', 'relu');
        output = temp.apply_spk_NL(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_spk_NL_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = test_case.net.set_params('spk_NL', 'sigmoid');
        output = temp.apply_spk_NL(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_spk_NL_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = test_case.net.set_params('spk_NL', 'softplus');
        output = temp.apply_spk_NL(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_spk_NL_deriv(data);
        test_case.verifySize(output, size(data))
    end
    
end
    
end