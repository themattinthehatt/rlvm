% TODO much later in the future
%   - reg matrices for stim subunits (only l2 supported now for fitting)
%   - fit_alt - break data into smaller time chunks, fit, stitch together
%   - fit_alt - make num_iters/verbose options in optim_params?
%   - unit testing for fit_weights/fit_latent_states/fit_alt
%   - turn 'fit_subs' option to fit_model into a field of fit_params?
%   - turn auto part into column-major temporal rep
%
%
%% ---------------------- VERSION 2.0 UPDATES -----------------------------
%
% Layer class
%   - (DONE 11/05/16) implement basic layer class
%   - add regularization option for external input weights
%
% RLVM
%   - (DONE 11/05/16) fit_model - single and stacked autoencoder models
%   - (DONE 11/06/16) fit_model - shared stim subunits with auto models
%   - fit_model - get individual stim subunits working with auto models
%   - set_reg_params - add option for external input weights
%   - add iters and func_evals to fit_history struct
%   - add pretraing to fit_params struct
%   - pretrain - replace Autoencoder with RLVM model
%
% Other



%% ---------------------- VERSION 1.0 UPDATES -----------------------------
%
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
%   - reorganize home directory
%       - (DONE 07/01/16) make @StimSubunit and @AutoSubunit directories
%       - (DONE 07/13/16) make data/doc/lib/src/test directories