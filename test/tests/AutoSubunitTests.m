classdef AutoSubunitTests < matlab.unittest.TestCase

% unit testing for AutoSubunit class methods
% TOFIX
% - set_params; doesn't handle user input when changing weight-tying
% - sort_clusters tests

properties 
    num_cells = 8;
    num_hid_nodes = 2;
    auto
end

methods (TestMethodSetup)

    function problem_setup(test_case)
        test_case.auto = AutoSubunit(test_case.num_cells, test_case.num_hid_nodes, 'gaussian');
        % normally inherited from RLVM
        test_case.auto.min_pred_rate = 1e-50; 
        test_case.auto.max_g = 50; 
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
        test_case.fatalAssertEqual(test_case.auto.num_cells, test_case.num_cells)
        test_case.fatalAssertEqual(test_case.auto.num_hid_nodes, test_case.num_hid_nodes)
    end
        
    function test_AutoSubunit(test_case)
        % test constructor
        test_case.verifyClass(AutoSubunit(), 'AutoSubunit')
        test_case.verifyError(@() AutoSubunit(1, 1), 'MATLAB:minrhs')
    end

    function test_setting_methods(test_case)
        % set_params
        temp = test_case.auto.set_params('num_cells', 30, ...
                                         'num_hid_nodes', 3, ...
                                         'act_func_hid', 'relu');
        test_case.verifyEqual(temp.num_cells, 30)
        test_case.verifyEqual(temp.num_hid_nodes, 3)
        test_case.verifySubstring(temp.act_func_hid, 'relu')
                
        % set_reg_params
        temp = test_case.auto.set_reg_params('l2_biases', 1, ...
                                             'l2_weights', 1, ... 
                                             'l1_hid', 1, ...
                                             'd2t_hid', 1);
        test_case.verifyEqual(temp.reg_lambdas.l2_biases1, 1)
        test_case.verifyEqual(temp.reg_lambdas.l2_biases2, 1)
        test_case.verifyEqual(temp.reg_lambdas.l2_weights1, 1)
        test_case.verifyEqual(temp.reg_lambdas.l2_weights2, 1)
        test_case.verifyEqual(temp.reg_lambdas.l1_hid, 1)
        test_case.verifyEqual(temp.reg_lambdas.d2t_hid, 1)
        
        temp = test_case.auto.set_reg_params('l2_biases1', 10, ...
                                             'l2_biases2', 10, ...
                                             'l2_weights1', 10, ...
                                             'l2_weights2', 10);
        test_case.verifyEqual(temp.reg_lambdas.l2_biases1, 10)
        test_case.verifyEqual(temp.reg_lambdas.l2_biases2, 10)
        test_case.verifyEqual(temp.reg_lambdas.l2_weights1, 10)
        test_case.verifyEqual(temp.reg_lambdas.l2_weights2, 10)
        
        % set_hidden_order
        data = randn(100, test_case.auto.num_cells);
        temp = test_case.auto.set_hidden_order(data);
        hidden_act = temp.get_model_internals(data);
        activity_stddev = std(hidden_act{1});
        activity_diff = diff(activity_stddev);
        test_case.verifyLessThanOrEqual(activity_diff, zeros(test_case.auto.num_hid_nodes - 1));
        
        % set_init_weights
        % wrapper function for AutoSubunit.set_init_weights_stat
        
    end

    function test_getting_methods(test_case)
        % get_model_internals
        data = randn(100, test_case.auto.num_cells);
        [fgint, gint] = test_case.auto.get_model_internals(data, 'indx_tr', 1:10);
        test_case.verifySize(gint{1}, [10, test_case.auto.num_hid_nodes])
        test_case.verifySize(fgint{1}, [10, test_case.auto.num_hid_nodes])
        test_case.verifySize(gint{2}, [10, test_case.auto.num_cells])
        
        % get_reg_pen
        temp = test_case.auto.set_reg_params('l2_biases', 1, 'l2_weights', 1, 'l1_hid', 1, 'd2t_hid', 1);
        reg_pen = temp.get_reg_pen(data);
        test_case.verifyGreaterThan(reg_pen.l2_weights1, 0)
        test_case.verifyGreaterThan(reg_pen.l2_weights2, 0)
        test_case.verifyGreaterThanOrEqual(reg_pen.l2_biases1, 0)
        test_case.verifyGreaterThanOrEqual(reg_pen.l2_biases2, 0)
        test_case.verifyGreaterThan(reg_pen.l1_hid, 0)
        test_case.verifyGreaterThan(reg_pen.d2t_hid, 0)
        
        % get_sparse_penalty
        data = randn(100, test_case.auto.num_hid_nodes);
        
        temp = test_case.auto.set_params('act_func_hid', 'lin');
        [func, grad] = temp.get_sparse_penalty(data);
        test_case.verifySize(func, [1, 1])
        test_case.verifySize(grad, [test_case.auto.num_hid_nodes, 1])
        
        temp = temp.set_params('act_func_hid', 'relu');
        [func, grad] = temp.get_sparse_penalty(data);
        test_case.verifySize(func, [1, 1])
        test_case.verifySize(grad, [test_case.auto.num_hid_nodes, 1])
        
        temp = temp.set_params('act_func_hid', 'sigmoid');
        [func, grad] = temp.get_sparse_penalty(data);
        test_case.verifySize(func, [1, 1])
        test_case.verifySize(grad, [test_case.auto.num_hid_nodes, 1])
        
        % get_weights
        temp = AutoSubunit(test_case.num_cells, test_case.num_hid_nodes, ...
                'gaussian', [], 0); % no weight tie
        weight_vec = randn(2 * test_case.num_cells * test_case.num_hid_nodes ...
                           + test_case.num_cells + test_case.num_hid_nodes, 1);
        [w1, w2, b1, b2] = temp.get_weights(weight_vec);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        
        temp = AutoSubunit(test_case.num_cells, test_case.num_hid_nodes, ...
                'gaussian', [], 1); % weight tie
        weight_vec = randn(test_case.num_cells * test_case.num_hid_nodes ...
                           + test_case.num_cells + test_case.num_hid_nodes, 1);
        [w1, w2, b1, b2] = temp.get_weights(weight_vec);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        
        % get_decoding_weights
        temp = AutoSubunit(test_case.num_cells, test_case.num_hid_nodes, ...
                'gaussian', [], 0);
        weight_vec = randn(test_case.num_cells * test_case.num_hid_nodes ...
                           + test_case.num_cells, 1);
        [w2, b2] = temp.get_decoding_weights(weight_vec);
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b2, [test_case.num_cells, 1])
    end

    function test_hidden_methods(test_case)
        % apply_act_func/apply_act_deriv
        data = randn(100, test_case.auto.num_cells);
        
        temp = test_case.auto.set_params('act_func_hid', 'lin');
        output = temp.apply_act_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_act_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = temp.set_params('act_func_hid', 'relu');
        output = temp.apply_act_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_act_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = temp.set_params('act_func_hid', 'sigmoid');
        output = temp.apply_act_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_act_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp = temp.set_params('act_func_hid', 'softplus');
        output = temp.apply_act_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_act_deriv(data);
        test_case.verifySize(output, size(data))
    end
    
    function test_static_methods(test_case)
        % set_init_weights_stat
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            'gaussian', test_case.num_cells, test_case.num_hid_nodes, 0);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            'gaussian', test_case.num_cells, test_case.num_hid_nodes, 1);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        test_case.verifyEqual(w1, w2')
        
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            'uniform', test_case.num_cells, test_case.num_hid_nodes, 0);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            'uniform', test_case.num_cells, test_case.num_hid_nodes, 1);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        test_case.verifyEqual(w1, w2')
        
        init_weights = randn(2 * test_case.num_cells * test_case.num_hid_nodes ...
                           + test_case.num_cells + test_case.num_hid_nodes, 1);
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            init_weights, test_case.num_cells, test_case.num_hid_nodes, 0);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        
        init_weights = randn(test_case.num_cells * test_case.num_hid_nodes ...
                           + test_case.num_cells + test_case.num_hid_nodes, 1);
        [w1, w2, b1, b2] = AutoSubunit.set_init_weights_stat( ...
            init_weights, test_case.num_cells, test_case.num_hid_nodes, 1);
        test_case.verifySize(w1, [test_case.num_cells, test_case.num_hid_nodes])
        test_case.verifySize(w2, [test_case.num_hid_nodes, test_case.num_cells])
        test_case.verifySize(b1, [test_case.num_hid_nodes, 1])
        test_case.verifySize(b2, [test_case.num_cells, 1])
        test_case.verifyEqual(w1, w2')
    end
    
end
    
end