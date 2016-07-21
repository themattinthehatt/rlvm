% VERSION 1.0 UPDATES
% StimSubunit
%   - (DONE 07/20/16) add shared stim subunit
%   - (DONE 07/12/16) abstract set_init_filt_stat away from ind or sh type
%   - (DONE 07/13/16) fix regularization normalization in get_reg_pen
%   - (DONE 07/12/16) create_tikhonov_matrix - get rid of stim_params input
%   - (DONE 07/12/16) make set_params method for mod_sign, NL_type, etc.
%
% AutoSubunit
%   - (DONE 07/13/16) make sortClusters function a static method
%   - (DONE 07/13/16) get_reg_pen: fix regularization normalization
%   - (DONE 07/14/16) check_inputs 'called by'
%
% RLVM
%   - (DONE 07/12/16) combine fitting methods
%   - (DONE 07/12/16) fit_weights: fix reg normalization in fit_weights
%   - (DONE 07/12/16) maxIter -> max_iter  
%   - (DONE 07/12/16) get rid of stim_params property (unnecessary)
%   - (DONE 07/13/16) get_eval_model: make Xstims optional argument
%   - (DONE 07/13/16) get_eval_model: fix regularization normalization
%   - (DONE 07/13/16) get_eval_model: pseudo-r2 for exp and poiss losses
%   - (DONE 07/13/16) fit_weights: redo check_inputs
%   - (DONE 07/14/16) fit_model - check for minFunc
%   - (DONE 07/19/16) fit_model - allow model subunits to be held constant
%   - (DONE 07/20/16) fit_model - optimize shared subunit func/grad calcs
%   - (DONE 07/20/16) display_stim_filts - now works for shared subunits
%
% Unit Testing
%   - (DONE 07/12/16) RLVM
%   - (DONE 07/12/16) AutoSubunit
%   - (DONE 07/12/16) StimSubunit
%   - (DONE 07/14/16) fit_weights_latent_states/auto
%   - (DONE 07/14/16) fit_weights_latent_states/stim_individual
%   - (DONE 07/14/16) fit_weights_latent_states/auto+stim_individual
%   - (DONE 07/20/16) fit_weights_latent_states/stim_shared
%   - (DONE 07/20/16) fit_weights_latent_states/auto+stim_shared
%
% Formatting/Organization
%   - reformat (indents are spaces, no tabs; add whitespace around args)
%       - (DONE 06/30/16) RLVM
%       - (DONE 07/11/16) StimSubunit
%       - (DONE 06/30/16) AutoSubunit
%       - (DONE 07/12/16) fit_params
%   - reformat method documentation
%       - (DONE 06/30/16) RLVM
%       - (DONE 07/11/16) StimSubunit
%       - (DONE 06/30/16) AutoSubunit
%       - (DONE 07/12/16) fit_params
%   - add more documentation at the top of class defs (examples?)
%       - RLVM
%       - AutoSubunit
%       - StimSubunit
%   - reorganize home directory
%       - (DONE 07/01/16) make @StimSubunit and @AutoSubunit directories
%       - (DONE 07/13/16) make data/doc/lib/src/test directories
%
%
% TODO much later in the future
%   - reg matrices for stim subunits (only l2 supported now for fitting)
%   - perhaps add a regpath to the auto model (but not stim+auto models)
%   - put in a lib folder with mark schmidt's minFunc
%   - add stim subunits to fit_weights and fit_latent_vars
%   - fit_alt - break data into smaller time chunks, fit, stitch together
%   - fit_alt - make num_iters/verbose options in optim_params?
%   - unit testing for fit_weights/fit_latent_states/fit_alt
%   - turn 'fit_subs' option to fit_model into a field of fit_params?
%
%
% VERSION 2.0 UPDATES
%   - refactor to have StimSubunit, IntLayer (integration layer) and Layer 
%     classes
%   - have LatentLayer be a Layer object, but named/handled differently by 
%     the RLVM class
%   - move to RLVM properties/methods?
%       - weight_tie
%       - set_hidden_order
%       - sort_clusters
%       - check_model_dims (run through layers, out-dim(k) <=> in-dim(k+1)
%   - have 'get_model_internals' method for Layer, or just do forward pass
%     in RLVM?
%   - forget sparsity
%
%   - IntLayer (inherits from Layer?)
%       - Properties
%           ** hid_weights
%           ** stim_weights
%           * biases
%           * reg_lambdas
%           * act_func
%           * num_in_nodes
%           * num_out_nodes (num_cells or something else)
%       - Methods
%           * IntLayer()
%           * set_act_func
%           * set_reg_params
%           * set_init_weights
%           * get_reg_pen
%           * get_weights (override Layer method, 2 sets)
%           * apply_act_func
%           * apply_act_deriv
%           * init_reg_lambdas (override Layer method, 2 sets)
%           * init_reg_lambdas
%           * set_init_weights_stat (override Layer method, 2 sets)
%
%   - Layer
%       - Properties
%           * weights
%           * biases
%           * reg_lambdas
%           * act_func
%           * num_in_nodes
%           * num_out_nodes
%       - Methods
%           * Layer()
%           * set_act_func
%           * set_reg_params
%           * set_init_weights
%           * get_reg_pen
%           * get_weights
%           * apply_act_func
%           * apply_act_deriv
%           * init_reg_lambdas
%           * set_init_weights_stat