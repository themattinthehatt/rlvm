classdef StimSubunitTests < matlab.unittest.TestCase

% unit testing for StimSubunit class methods
% TOFIX
%   - multiple subunits
%   - more complicated stim models
%   - create_tikhonov_matrix - doesn't handle split points
%   - doesn't test split_pts

properties 
    num_lags = 6;
    num_xpix = 4;
    num_ypix = 2;
    num_cells = 8;
    stim
    expt_len = 100;
    xstim
end

methods (TestMethodSetup)

    function problem_setup(test_case)
        stim_params = StimSubunit.create_stim_params( ...
            [test_case.num_lags, test_case.num_xpix, test_case.num_ypix], ...
            test_case.num_cells);
        test_case.xstim{1} = StimSubunit.create_time_embedding(...
            randn(test_case.expt_len, test_case.num_xpix*test_case.num_ypix), ...
            stim_params);
        test_case.stim = StimSubunit(stim_params, 'gaussian', 1, 'lin', 1);
        % normally inherited from RLVM
        test_case.stim.min_pred_rate = 1e-50; 
        test_case.stim.max_g = 50; 
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
        test_case.fatalAssertEqual(test_case.stim.stim_params.dims, ...
            [test_case.num_lags, test_case.num_xpix, test_case.num_ypix])
        test_case.fatalAssertEqual(test_case.stim.stim_params.num_outputs, ...
            test_case.num_cells)
        test_case.fatalAssertSize(test_case.stim.filt, ...
            [prod(test_case.stim.stim_params.dims), test_case.num_cells])
        test_case.fatalAssertSize(test_case.xstim{1}, ...
            [test_case.expt_len, prod(test_case.stim.stim_params.dims)])
    end
        
    function test_StimSubunit(test_case)
        % test constructor
        test_case.verifyClass(StimSubunit(), 'StimSubunit')
        test_case.verifyError(@() StimSubunit(1, 1), 'MATLAB:minrhs')
    end

    function test_setting_methods(test_case)
        % set_stim_params
        temp = test_case.stim.set_stim_params('dims', [1, 2, 3], ...
                                              'tent_spacing', 3, ...
                                              'boundary_conds', [0, 0, 0]);
        test_case.verifyEqual(temp.stim_params.dims, [1, 2, 3])
        test_case.verifyEqual(temp.stim_params.tent_spacing, 3)
        test_case.verifyEqual(temp.stim_params.boundary_conds, [0, 0, 0])
        
        % set_reg_params
        test_case.stim = test_case.stim.set_reg_params('l2', 1, 'd2t', 1, ...
                                                       'd2x', 1, 'd2xt', 1);
        test_case.verifyEqual(test_case.stim.reg_lambdas.l2, 1)
        test_case.verifyEqual(test_case.stim.reg_lambdas.d2t, 1)
        test_case.verifyEqual(test_case.stim.reg_lambdas.d2x, 1)
        test_case.verifyEqual(test_case.stim.reg_lambdas.d2xt, 1)
        
        % set_init_filt
        % wrapper function for StimSubunit.set_init_filt_stat
    end
    
    function test_getting_methods(test_case)
        % get_model_internals
        [fgint, gint] = test_case.stim.get_model_internals( ...
                            test_case.xstim, 'indx_tr', 1:10);
        test_case.verifySize(gint, [10, test_case.num_cells])
        test_case.verifySize(fgint, [10, test_case.num_cells])
        
        % get_reg_pen
        test_case.stim = test_case.stim.set_reg_params('l2', 1, 'd2t', 1, ...
                                                       'd2x', 1, 'd2xt', 1);
        reg_pen = test_case.stim.get_reg_pen();
        test_case.verifyGreaterThan(reg_pen.l2, 0)
        test_case.verifyGreaterThan(reg_pen.d2t, 0)
        test_case.verifyGreaterThan(reg_pen.d2x, 0)
        test_case.verifyGreaterThan(reg_pen.d2xt, 0)
    end
    
    function test_hidden_methods(test_case)
        % apply_NL_func/apply_NL_deriv
        data = randn(test_case.expt_len, test_case.num_cells);
        temp = test_case.stim;
        
        temp.NL_type = 'lin';
        output = temp.apply_NL_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_NL_deriv(data);
        test_case.verifySize(output, size(data))
        
        temp.NL_type = 'relu';
        output = temp.apply_NL_func(data);
        test_case.verifySize(output, size(data))
        output = temp.apply_NL_deriv(data);
        test_case.verifySize(output, size(data))
        
        % make_tikhonov_matrices
        test_case.stim = test_case.stim.set_reg_params('l2', 1, 'd2t', 1, ...
                                                       'd2x', 1, 'd2xt', 1);
        tmats = test_case.stim.make_tikhonov_matrices();
        filt_size = prod(test_case.stim.stim_params.dims);
        test_case.verifySize(tmats(1).tmat, [filt_size, filt_size])
        test_case.verifySize(tmats(2).tmat, [filt_size, filt_size])
        test_case.verifySize(tmats(3).tmat, [filt_size, filt_size])
        
        % create_tikhonov_matrix
        % temporal component only
        temp = test_case.stim;
        temp.stim_params = StimSubunit.create_stim_params([10, 1, 1], 1);
        test_case.verifySize(temp.create_tikhonov_matrix('d2t'), [10, 10]);
        test_case.verifyError(@() temp.create_tikhonov_matrix('d2x'), ...
            'StimSubunit:invalidoption');
        test_case.verifyError(@() temp.create_tikhonov_matrix('d2xt'), ...
            'StimSubunit:invalidoption');
        
        % single spatial component only
        temp.stim_params = StimSubunit.create_stim_params([1, 10, 1], 1);
        test_case.verifySize(temp.create_tikhonov_matrix('d2t'), [10, 10]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2x'), [10, 10]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2xt'), [10, 10]);
        
        % temporal component and single spatial component
        temp.stim_params = StimSubunit.create_stim_params([10, 10, 1], 1);
        test_case.verifySize(temp.create_tikhonov_matrix('d2t'), [100, 100]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2x'), [100, 100]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2xt'), [100, 100]);
        
        % temporal component and two spatial components
        temp.stim_params = StimSubunit.create_stim_params([10, 10, 10], 1);
        test_case.verifySize(temp.create_tikhonov_matrix('d2t'), [1000, 1000]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2x'), [1000, 1000]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2xt'), [1000, 1000]);
        
        % two spatial components
        temp.stim_params = StimSubunit.create_stim_params([1, 10, 10], 1);
        test_case.verifySize(temp.create_tikhonov_matrix('d2t'), [100, 100]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2x'), [100, 100]);
        test_case.verifySize(temp.create_tikhonov_matrix('d2xt'), [100, 100]);
    end
    
    function test_static_methods(test_case)
        % set_init_filt_stat
        stim_params = StimSubunit.create_stim_params([test_case.num_lags ...
                                                      test_case.num_xpix ...
                                                      test_case.num_ypix], ...
                                                      test_case.num_cells);
        filt = StimSubunit.set_init_filt_stat(stim_params, 'gaussian');
        test_case.verifySize(filt, ...
            [prod(test_case.stim.stim_params.dims), test_case.num_cells])
        filt = StimSubunit.set_init_filt_stat(stim_params, 'uniform');
        test_case.verifySize(filt, ...
            [prod(test_case.stim.stim_params.dims), test_case.num_cells])
        filt = StimSubunit.set_init_filt_stat(stim_params, randn( ...
            prod(test_case.stim.stim_params.dims), test_case.num_cells));
        test_case.verifySize(filt, ...
            [prod(test_case.stim.stim_params.dims), test_case.num_cells])
        
        % create_stim_params
        stim_params = StimSubunit.create_stim_params( ...
            [test_case.num_lags, test_case.num_xpix, test_case.num_ypix], ...
            test_case.num_cells, ...
            'stim_dt', 2, ...
            'up_fac', 3, ...
            'tent_spacing', 4, ...
            'boundary_conds', [0, Inf, -1], ...
            'split_pts', [5, 6, 7]);
        test_case.verifyEqual(stim_params.dims, ...
            [test_case.num_lags, test_case.num_xpix, test_case.num_ypix]);
        test_case.verifyEqual(stim_params.dt, 2/3, 'RelTol', sqrt(eps))
        test_case.verifyEqual(stim_params.up_fac, 3)
        test_case.verifyEqual(stim_params.tent_spacing, 4)
        test_case.verifyEqual(stim_params.boundary_conds, [0, Inf, -1])
        test_case.verifyEqual(stim_params.split_pts, [5, 6, 7])
        test_case.verifyEqual(stim_params.num_outputs, test_case.num_cells)
        
        % create_time_embedding
        % temporal only
        stim_vec = randn(test_case.expt_len, 1);
        stim_params = StimSubunit.create_stim_params([10, 1, 1], 3);
        xmat_1 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_1, [test_case.expt_len, 10])
        
        stim_params = StimSubunit.create_stim_params([5, 1, 1], 3, ...
            'tent_spacing', 2);
        xmat_2 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_2, [test_case.expt_len, 5])
        
        test_case.verifyNotEqual(xmat_1, xmat_2)
        
        % temporal and one spatial dim
        stim_vec = randn(test_case.expt_len, 4);
        stim_params = StimSubunit.create_stim_params([10, 4, 1], 3);
        xmat_1 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_1, [test_case.expt_len, 10 * 4])
        
        stim_params = StimSubunit.create_stim_params([5, 4, 1], 3, ...
            'tent_spacing', 2);
        xmat_2 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_2, [test_case.expt_len, 5 * 4])
        
        test_case.verifyNotEqual(xmat_1, xmat_2)
        
        % temporal and two spatial dims
        stim_vec = randn(test_case.expt_len, 4, 6);
        stim_params = StimSubunit.create_stim_params([10, 4, 6], 3);
        xmat_1 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_1, [test_case.expt_len, 10 * 4 * 6])
        
        stim_params = StimSubunit.create_stim_params([5, 4, 6], 3, ...
            'tent_spacing', 2);
        xmat_2 = StimSubunit.create_time_embedding(stim_vec, stim_params);
        test_case.verifySize(xmat_2, [test_case.expt_len, 5 * 4 * 6])
        
        test_case.verifyNotEqual(xmat_1, xmat_2)
    end
    
    function test_static_hidden_methods(test_case)
        % shift_mat_zpad
        x = randn(6);
        xshift = StimSubunit.shift_mat_zpad(x, 1, 1);
        test_case.verifyEqual(x(1:end-1,:), xshift(2:end,:))
        xshift = StimSubunit.shift_mat_zpad(x, -1, 1);
        test_case.verifyEqual(x(2:end,:), xshift(1:end-1,:))
        xshift = StimSubunit.shift_mat_zpad(x, 1, 2);
        test_case.verifyEqual(x(:,1:end-1), xshift(:,2:end))
        xshift = StimSubunit.shift_mat_zpad(x, -1, 2);
        test_case.verifyEqual(x(:,2:end), xshift(:,1:end-1))
    end
    
end

end