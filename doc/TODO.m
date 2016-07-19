% VERSION 1.0 UPDATES
% StimSubunit
%   - add shared stim subunit
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
%   - fit_model - let some model components be held constant
%   - fit_alt - make num_iters/verbose options in optim_params?
%
% Unit Testing
%   - (DONE 07/12/16) RLVM
%   - (DONE 07/12/16) AutoSubunit
%   - (DONE 07/12/16) StimSubunit
%   - (DONE 07/14/16) fit_weights_latent_states/auto
%   - (DONE 07/14/16) fit_weights_latent_states/stim_individual
%   - (DONE 07/14/16) fit_weights_latent_states/auto+stim_individual
%   - fit_weights_latent_states/stim_shared
%   - fit_weights_latent_states/auto+stim_shared
%   - fit_weights
%   - fit_latent_states
%   - fit_alt
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
%   - add more documentation at the top of class def (examples?)
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
%   - is it possible to speed up fit_ls, at least in the gaussian case?
%   - add alt fitting routine - make autoencoder one way to initialize
%   - display_stim_filters: move into StimSubunit
%   - display_stim_filters: display multiple cells properly
%   - if more than one hidden layer is used, some major changes will need
%     to take place; some can occur preemptively, e.g. instead of
%     act_func_hid and act_func_out, have act_funcs be a cell array of
%     strings, then call things like apply_act_func(stuff, act_funcs{1})
%   - perhaps add a regpath to the auto model (but not stim+auto models)
%   - maybe look at optimizing the objective function implementation(s)
%   - put in a lib folder with mark schmidt's minFunc
%   - generalize fit_weights and fit_latent_vars