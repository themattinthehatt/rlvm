classdef StimSubunit
    
% Class implementing the stimulus subunits of an RLVM object
%
% Reference:
%
% Author: Matt Whiteway
%   07/11/16

properties
    filt            % vector or matrix of filter coefficients
    offsets         % scalar or vector for bias term(s), before NL
    mod_sign        % +/-1, defines subunit as excitatory or inhibitory
    NL_type         % type of nonlinearity on filter output
    x_target        % specifies which Xmat filter acts on
    reg_lambdas     % struct defining reg params for various types of reg
    stim_params     % struct defining stim params for x_target
    init_params     % struct saving initialization parameters
      % rng_state
      % stim_init_filt
end

properties (Hidden) % inherited from RLVM
    allowed_stim_regtypes = {'l2', 'd2t', 'd2x', 'd2xt'};    
    allowed_stim_NLtypes  = {'lin', 'relu', 'softplus'};
    min_pred_rate
    max_g
end

%% ********************  constructor **************************************
methods
      
    function subunit = StimSubunit(stim_params, init_filt, mod_sign, ...
                                   NL_type, x_target)
    % subunit = StimSubunit(stim_params, init_filt, mod_sign, ...
    %                              NL_type, num_outputs, <x_target>)
    % 
    % Constructor for StimSubunit class
    %
    % INPUTS:
    %   stim_params:    stim_params struct from StimSubunit static method
    %   init_filt:      string specifying a random initialization 
    %                   ('gaussian' or 'uniform') or a matrix of the
    %                   appropriate size for the intended x_target
    %                   [filt_len x 1] or [filt_len x num_cells]
    %   mod_sign:       +/-1, defines subunit as excitatory or inhibitory
    %   NL_type:        string specifying type of nonlinearity that filter
    %                   output passes through
    %   <x_target>:     optional; scalar index specifying which stimulus 
    %                   element this subunit acts on
    %
    % OUTPUTS:
    %   subunit:        initialized StimSubunit object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return 
    end
    
    % define defaults
    if (nargin < 5 || isempty(x_target))
        x_target = 1;
    end
    
    % parse inputs
    assert(ismember(mod_sign, [-1, 1]), ...
        'Invalid mod_sign')
    assert(ismember(NL_type, subunit.allowed_stim_NLtypes), ...
        'Invalid NL_type')
    
    % initialize weights
    [filt_, init_param_struct] = StimSubunit.set_init_filt_stat( ...
                                    stim_params, init_filt);
    
    % set properties
    subunit.filt = filt_;
    subunit.offsets = []; % for relu; currently not supported
    subunit.mod_sign = mod_sign;
    subunit.NL_type = NL_type;
    subunit.x_target = x_target;
    subunit.reg_lambdas = StimSubunit.init_reg_lambdas(); % init all to 0s
    subunit.stim_params = stim_params;
    subunit.init_params = init_param_struct;
    
    end % method
    
end
%% ********************  setting methods **********************************
methods
    
    function subunit = set_stim_params(subunit, varargin)
    % subunit = subunit.set_stim_params(kargs)
    %
    % Takes a sequence of key-value pairs to set stim parameters for
    % StimSubunit object. Note that without access to the Xstims cell array
    % there is no way to check that these changes are compatible.
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'x_target', scalar
    %           stimulus target to apply new stim params to
    %       'dims', 1x3 vector 
    %           [num_lags, num_xpix, num_ypix]
    %           defines dimensionality of the (time-embedded) stimulus. If
    %           no x_target is specified, defaults to 1. 
    %       'tent_spacing', scalar
    %           optional spacing of tent-basis functions when using a 
    %           tent-basis representaiton of the stimulus. Allows for the 
    %           stimulus filters to be represented at a lower time 
    %           resolution than other model components. 
    %       'boundary_conds', 1x3 vector
    %           vector of boundary conditions on each dimension 
    %           Inf is free, 0 is tied to 0, and -1 is periodic
    %       'split_pts', 1x3 vector 
    %           [direction boundary_ind boundary_cond]
    %           specifies an internal boundary as a 3-element vector
    %
    % OUTPUTS:
    %   subunit: updated StimSubunit object

    % check for appropriate number of inputs
    assert(mod(length(varargin),2) == 0, ...
        'Input should be a list of key-value pairs')

    % parse inputs
    i = 1; 
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'x_target'
                subunit.x_target = varargin{i+1};
            case 'dims'
                subunit.stim_params.dims = varargin{i+1};
                % pad stim_dims with 1's for bookkeeping
                while length(subunit.stim_params.dims) < 3
                    subunit.stim_params.dims = ...
                    cat(2, subunit.stim_params.dims, 1); 
                end
            case 'tent_spacing'
                subunit.stim_params.tent_spacing = varargin{i+1};
            case 'boundary_conds'
                assert(all(ismember(varargin{i+1}, [-1, 0, Inf])), ...
                    'Incorrect boundary condition specification')
                subunit.stim_params.boundary_conds = varargin{i+1};
            case 'split_pts'
                subunit.stim_params.split_pts = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
   
    end % method
    
    
    function subunit = set_reg_params(subunit, varargin)
    % subunit = subunit.set_reg_params(kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters
    % for the StimSubunit object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'reg_type', scalar
    %       reg_types:
    %       'l2' | 'd2t' | 'd2x' | 'd2xt'
    %   
    % OUTPUTS:
    %   subunit: updated StimSubunit object
    
    % check for appropriate number of inputs
    assert(mod(length(varargin), 2) == 0, ...
        'Input should be a list of key-value pairs')
    
    % parse inputs
    i = 1;
    reg_types = {}; reg_vals = [];
    while i <= length(varargin)
        switch lower(varargin{i})
            case subunit.allowed_stim_regtypes
                reg_types = cat(1,reg_types,lower(varargin{i}));
                % build up a [1xP] matrix; P is the number of reg types
                reg_vals = cat(2,reg_vals,varargin{i+1}); 
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % check for actual input
    if isempty(reg_vals)
        warning('No regularization values specified, no action taken');
    end
    
    % apply regs
    assert(all(reg_vals >= 0), ...
        'regularization hyperparameters must be non-negative');
    for i = 1:length(reg_types)
        subunit.reg_lambdas.(reg_types{i}) = reg_vals(i);
    end
    
    end % method
  
    
    function subunit = set_init_filt(subunit, init_filt)
    % subunit = subunit.set_init_weights(init_filt)
    %
    % Sets filt property of StimSubunit object
    %
    % INPUTS:
    %   init_weights:   either a string specifying type of random 
    %                   initialization for weights ('gaussian' or
    %                   'uniform') or a weight vector of appropriate length  
    %
    % OUTPUT:
    %   subunit:        updated StimSubunit object
    
    % call static set_init_weights used in Constructor
    [filt_,init_params_struct] = ...
    StimSubunit.set_init_filt_stat(subunit.stim_params,init_filt);

    % set properties
    subunit.filt = filt_;
    subunit.init_params = init_params_struct;
    
    end % method
    
end
%% ********************  getting methods **********************************
methods
    
    function [fgint, gint] = get_model_internals(subunit, Xstims, varargin)
    % subunit = subunit.get_model_internals(pop_activity, Xstims, kargs)
    %
    % Retrieves informaton from stimulus models
    %
    % INPUTS:
    %   Xstims:     num_stims x 1 cell array with T x _ matrix of stim 
    %               values
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation
    %
    % OUTPUTS:
    %   fgint: T x 1 vector or T x num_cells matrix of subunit outputs       
    %   gint:  T x 1 vector or T x num_cells matrix of filter outputs
    
    % check inputs (filter dimensions)
    assert(size(Xstims{subunit.x_target}, 2) == size(subunit.filt, 1), ...
        'Xstims dims inconsistent with stimulus filter')
    
    % define defaults
    indx_tr = NaN; % NaN means we use all available data
    
    % parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(Xstims{1}, 1))) ...
                    || isnan(varargin{i+1}), ...
                    'Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('Invalid input flag');
        end
        i = i + 2;
    end
    
    % use indx_tr
    if ~isnan(indx_tr)
        Xstims{subunit.x_target} = Xstims{subunit.x_target}(indx_tr,:);
    end
    
    % apply filters to stimulus
    gint = Xstims{subunit.x_target} * subunit.filt; 
    fgint = subunit.apply_NL_func(gint);
   
    end % method
    
    
    function reg_pen = get_reg_pen(subunit)
    % reg_pen = subunit.get_reg_pen()
    %
    % Retrieves regularization penalties on cost function of stimulus model
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_pen: struct containing penalties due to different regs

    tmats = subunit.make_tikhonov_matrices();

    % loop over the derivative regularization matrices
    for i = 1:length(tmats) 
        lambda = subunit.reg_lambdas.(tmats(i).type);
        reg_pen.(tmats(i).type) = 0.5 * lambda * sum(sum(...
            (tmats(i).tmat * subunit.filt).^2));
    end

    % get l2 penalties
    reg_pen.l2 = 0.5 * subunit.reg_lambdas.l2 * sum(sum(subunit.filt.^2));

    end % method
    
end
%% ********************  display methods **********************************
methods
    
    function [] = disp_filt(subunit, sub_num, varargin)
    % subunit.display_filt(<cell_num>, <weight>)
    %
    % Plots stimulus filter of the given subunit for the input cell number
    %
    % INPUTS:
    %   sub_num:    for titles
    %   cell_num:   optional; indices of cells to plot (used for plotting
    %               individual stim subunits)
    %   weight:     optional; weight to multiply filter by (used for
    %               plotting shared stim subunits)
    %
    % OUTPUTS:
    %   fig_handle: figure handle of output

    % check inputs
    if ~isempty(varargin)
        if length(varargin) == 2
            % cell index
            if ~isempty(varargin{1})
                assert(all(ismember(varargin{1}, 1:size(subunit.filt, 2))), ...
                    'Invalid cell index')
                filt_ = subunit.filt(:,varargin{1});
            else
                filt_ = subunit.filt;
            end
            % weight
            weight = varargin{2};
        elseif length(varargin) == 1
            % cell index
            assert(all(ismember(varargin{1}, 1:size(subunit.filt, 2))), ...
                'Invalid cell index')
            filt_ = subunit.filt(:,varargin{1});
            % weight
            weight = 1;
        else
            error('Too many inputs')
        end
    else
        filt_ = subunit.filt;
        weight = 1;
    end
    
    if subunit.stim_params.dims(1) == 1
        % only spatial component
        if prod(subunit.stim_params.dims(2:3)) == 1
            % only a single parameter
            stem(weight * filt_)
            title(sprintf('Subunit %i', sub_num), 'FontSize', 12)
            ylabel('param value')
        elseif subunit.stim_params.dims(3) == 1
            % one spatial dim
            plot(weight * filt_)
            title(sprintf('Subunit %i', sub_num), 'FontSize', 12)
            xlabel('x\_pix')
        else
            % two spatial dims
            filt_ = reshape(weight * filt_, subunit.stim_params.dims(2), ...
                                            subunit.stim_params.dims(3));
            imagesc(filt_, [-max(abs(filt_(:))), max(abs(filt_(:)))]);
            title(sprintf('Subunit %i', sub_num), 'FontSize', 12)
            xlabel('y\_pix')
            ylabel('x\_pix')
            set(gca,'YDir', 'normal')
            colormap(jet);
        end
    else
        % temporal component
        if prod(subunit.stim_params.dims(2:3)) == 1
            % only temporal component
            plot(weight * filt_)
            title(sprintf('Subunit %i', sub_num), 'FontSize', 12)
            xlabel('lags')
        elseif subunit.stim_params.dims(3) == 1
            % one spatial dim
            filt_ = reshape(weight * filt_, subunit.stim_params.dims(1), ...
                                            subunit.stim_params.dims(2));
            imagesc(filt_, [-max(abs(filt_(:))), max(abs(filt_(:)))]);
            title(sprintf('Subunit %i', sub_num), 'FontSize', 12)
            xlabel('x\_pix')
            ylabel('lags')
            set(gca,'YDir', 'normal')
            colormap(jet);
        else 
            % two spatial dims + temporal component
            error('Filter dimensions not currently supported for display')
        end
    end 
    
    end % method
    
end
%% ********************  hidden methods ***********************************
methods (Hidden)
    
    function sig = apply_NL_func(subunit, sig)
    % sig = subunit.apply_NL_func(sig)
    %
    % Applies the nonlinearity of the subunit to a given generating signal
    %
    % INPUTS:
    %   sig:    T x 1 vector that results from inner product of
    %           stimulus and filter
    %
    % OUTPUTS:
    %   sig:    T x 1 vector of generating signal that has been passed 
    %           through the subunit's activation function

    switch subunit.NL_type
        case 'lin'
        case 'relu'
            sig = max(0,sig);
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of under/overflow
            % appx linear
            temp_sig(sig > subunit.max_g) = sig(sig > subunit.max_g);
            % so LL isn't undefined
            temp_sig(temp_sig < subunit.min_pred_rate) = subunit.min_pred_rate;
            sig = temp_sig;
    end
    
    end % method
    
    
    function sig = apply_NL_deriv(subunit, sig)
    % sig = subunit.apply_NL_deriv(sig)
    %
    % Applies the derivative of the nonlinearity of the subunit to a given
    % generating signal
    %
    % INPUTS:
    %   sig:    T x 1 vector that results from inner product of
    %           stimulus and filter
    %
    % OUTPUTS:
    %   sig:    T x 1 vector of generating signal that has been passed 
    %           through the derivative of subunit's activation function

    switch subunit.NL_type
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            sig(sig<=0) = 0;
            sig(sig>0)  = 1;
        case 'softplus'
            temp_sig = exp(sig) ./ (1 + exp(sig));
            temp_sig(sig > subunit.max_g) = 1; % e^x/(1+e^x) => 1 for large x
            sig = temp_sig;
    end
    
    end % method
    
    
    function tmats = make_tikhonov_matrices(subunit)
    % tmats = subunit.make_tikhonov_matrices()
    %
    % Creates a struct containing the Tikhonov regularization matrices, 
    % given the stimulus and regularization parameters specified in the
    % subunit
    %
    % INPUTS:
    %   none
    % 
    % OUTPUTS:
    %   tmats:  struct array containing reg matrices and reg types

    % set of regularization types where we need a Tikhonov matrix
    deriv_reg_types = subunit.allowed_stim_regtypes( ...
        strncmp(subunit.allowed_stim_regtypes, 'd', 1)); 
    cnt = 1;
    tmats = [];
    
    % check each possible derivative regularization type
    for i = 1:length(deriv_reg_types) 
        lambda = subunit.reg_lambdas.(deriv_reg_types{i});
        if lambda > 0
            curr_tmat = subunit.create_tikhonov_matrix(deriv_reg_types{i});
            tmats(cnt).tmat = curr_tmat;
            tmats(cnt).type = deriv_reg_types{i};
            cnt = cnt + 1;
        end
    end 

    end % method
	
    
    function tmat = create_tikhonov_matrix(subunit, reg_type)
    % tmat = subunit.create_tikhonov_matrix(stim_params, reg_type)
    %
    % Creates a matrix specifying a an L2-regularization operator of the 
    % form ||T*k||^2, where T is the operator and k is a filter.
    %
    % INPUTS:
    %   stim_params: parameter struct associated with the target stimulus
    %       Fields:
    %           dims: specifies the number of stimulus elements along each 
    %                 dimension
    %           <boundary_conds>: specifies boundary conditions: Inf is 
    %                 free boundary, 0 is tied to 0, and -1 is periodic. 
    %           <.split_pts>: specifies an 'internal boundary' over which 
    %                 we dont smooth. [direction split_ind split_bnd]
    %   reg_type: string specifying type of regularization matrix to create
    %
    % OUTPUTS:
    %   tmat: sparse matrix specifying the desired Tikhonov operation
    %
    % The method of computing sparse differencing matrices used here is 
    % adapted from Bryan C. Smith's and Andrew V. Knyazev's function 
    % "laplacian", available here: 
    % http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d
    % This method is taken directly from the NIM code.

    % first dimension is assumed to represent time
    num_lags = subunit.stim_params.dims(1);			
    % additional dimensions are treated as spatial
    num_pix = squeeze(subunit.stim_params.dims(2:3));	
    allowed_reg_types = {'d2xt', 'd2x', 'd2t'};
    assert(ischar(reg_type) && ismember(reg_type, allowed_reg_types), ...
        'not an allowed regularization type');

    has_split = ~isempty(subunit.stim_params.split_pts);
    
    % check for temporal component
    if ismember(reg_type, {'d2xt', 'd2t'})
        
        et = ones(num_lags,1);
        if isinf(subunit.stim_params.boundary_conds(1)) 
            % if temporal dim has free boundary
            et([1 end]) = 0;
        end
    end
    
    % check for spatial component
    if ismember(reg_type, {'d2xt', 'd2x'})
        ex = ones(num_pix(1),1);
        if isinf(subunit.stim_params.boundary_conds(2)) 
            % if first spatial dim has free boundary
            ex([1 end]) = 0;
        end
        ey = ones(num_pix(2),1);
        if isinf(subunit.stim_params.boundary_conds(3)); 
            % if second spatial dim has free boundary
            ey([1 end]) = 0;
        end
    end

    % for 0-spatial-dimensional stimuli can only do temporal
    if num_pix == 1 

        assert(ismember(reg_type, {'d2t'}), ...
            'StimSubunit:invalidoption', ...
            'can only do temporal reg for stimuli without spatial dims');
        tmat = spdiags([et -2*et et], [-1 0 1], num_lags, num_lags);
        if subunit.stim_params.boundary_conds(1) == -1
            % if periodic boundary cond
            tmat(end,1) = 1; tmat(1,end) = 1;
        end
        if has_split
            assert(subunit.stim_params.split_pts(1) == 1, ...
                'check stim_params split_pts specification');
            tmat = StimSubunit.split_tmat(tmat, ...
                subunit.stim_params.split_pts);
        end

    % for 1-spatial dimensional stimuli
    elseif num_pix(2) == 1 
        
        if strcmp(reg_type, 'd2t') 
            % if temporal deriv
            D1t = spdiags([et -2*et et], [-1 0 1], num_lags, num_lags)';
            if subunit.stim_params.boundary_conds(1) == -1 
                % if periodic boundary cond
                D1t(end,1) = 1; D1t(1,end) = 1;
            end
            if has_split && subunit.stim_params.split_pts(1) == 1 
                % if theres a split along the time dim
                D1t = StimSubunit.split_tmat(D1t, ...
                    subunit.stim_params.split_pts); 
            end
            Ix = speye(num_pix(1));
            tmat = kron(Ix, D1t);
            
        elseif strcmp(reg_type, 'd2x')
            % if spatial deriv
            It = speye(num_lags);
            D1x = spdiags([ex -2*ex ex], [-1 0 1], num_pix(1), num_pix(1))';
            if subunit.stim_params.boundary_conds(2) == -1 
                % if periodic boundary cond
                D1x(end,1) = 1; D1x(1,end) = 1;
            end
            if has_split && subunit.stim_params.split_pts(1) == 2 
                % if theres a split along the spatial dim
                D1x = StimSubunit.split_tmat(D1x, ...
                    subunit.stim_params.split_pts); 
            end
            tmat = kron(D1x, It);
            
        elseif strcmp(reg_type, 'd2xt') 
            % if spatiotemporal laplacian
            D1t = spdiags([et -2*et et], [-1 0 1], num_lags, num_lags)';
            if subunit.stim_params.boundary_conds(1) == -1 
                % if periodic boundary cond
                D1t(end,1) = 1; D1t(1,end) = 1;
            end
            D1x = spdiags([ex -2*ex ex], [-1 0 1], num_pix(1), num_pix(1))';
            if subunit.stim_params.boundary_conds(2) == -1 
                % if periodic boundary cond
                D1x(end,1) = 1; D1x(1,end) = 1;
            end
            if has_split
                if subunit.stim_params.split_pts(1) == 1
                    D1t = StimSubunit.split_tmat(D1t, ...
                        subunit.stim_params.split_pts); 
                elseif subunit.stim_params.split_pts(1) == 2
                    D1x = StimSubunit.split_tmat(D1x, ...
                        subunit.stim_params.split_pts);
                else
                    error('invalid split dim');
                end
            end
        It = speye(num_lags);
        Ix = speye(num_pix(1));
        tmat = kron(Ix, D1t) + kron(D1x, It);
        end
    
    % for stimuli with 2-spatial dimensions
    else 
        if strcmp(reg_type, 'd2t') 
            % temporal deriv
            D1t = spdiags([et -2*et et], [-1 0 1], num_lags, num_lags)';
            if subunit.stim_params.boundary_conds(1) == -1
                % if periodic boundary cond
                D1t(end,1) = 1; D1t(1,end) = 1;
            end
            if has_split && subunit.stim_params.split_pts(1) == 1 
                % if splitting along temporal dim
                D1t = StimSubunit.split_tmat(D1t, ...
                    subunit.stim_params.split_pts); 
            end
            Ix = speye(num_pix(1));
            Iy = speye(num_pix(2));
            tmat = kron(Iy, kron(Ix, D1t));
            
        elseif strcmp(reg_type,'d2x')
            % spatial laplacian
            It = speye(num_lags);
            Ix = speye(num_pix(1));
            Iy = speye(num_pix(2));
            D1x = spdiags([ex -2*ex ex], [-1 0 1], num_pix(1), num_pix(1))';
            if subunit.stim_params.boundary_conds(2) == -1 
                % if periodic boundary cond
                D1x(end,1) = 1; D1x(1,end) = 1;
            end
            D1y = spdiags([ey -2*ey ey], [-1 0 1], num_pix(2), num_pix(2))';
            if subunit.stim_params.boundary_conds(3) == -1 
                % if periodic boundary cond
                D1y(end,1) = 1; D1y(1,end) = 1;
            end
            if has_split && ismember(subunit.stim_params.split_pts(1), [2 3])
                error('Cant do splits along spatial dims with 2-spatial dim stims yet'); 
            end
            tmat = kron(Iy, kron(D1x, It)) + kron(D1y, kron(Ix, It));
        
        elseif strcmp(reg_type,'d2xt')
            It = speye(num_lags);
            Ix = speye(num_pix(1));
            Iy = speye(num_pix(2));
            D1t = spdiags([et -2*et et], [-1 0 1], num_lags, num_lags)';
            if subunit.stim_params.boundary_conds(1) == -1 
                % if periodic boundary cond
                D1t(end,1) = 1; D1t(1,end) = 1;
            end
            D1x = spdiags([ex -2*ex ex], [-1 0 1], num_pix(1), num_pix(1))';
            if subunit.stim_params.boundary_conds(2) == -1 
                % if periodic boundary cond
                D1x(end,1) = 1; D1x(1,end) = 1;
            end
            D1y = spdiags([ey -2*ey ey], [-1 0 1], num_pix(2), num_pix(2))';
            if subunit.stim_params.boundary_conds(3) == -1 
                % if periodic boundary cond
                D1y(end,1) = 1; D1y(1,end) = 1;
            end
            if has_split
                if subunit.stim_params.split_pts(1) == 1
                    D1t = StimSubunit.split_tmat(D1t, ...
                        subunit.stim_params.split_pts);
                elseif ismember(subunit.stim_params.split_pts(1), [2 3])
                    error('Cant do splits along spatial dims with 2-spatial dim stims yet'); 
                end
            end
            tmat = kron(D1y, kron(Ix, It)) + ...
                   kron(Iy, kron(D1x, It)) + ...
                   kron(Iy, kron(Ix, D1t));
        end
    end

    end % method

    
    function check_inputs(subunit, Xstims)
    % subunit.check_inputs(Xstims)
    %
    % Checks if the parameters of subunit are consistent with the given
    % input
    %
    % INPUTS:
    %   pop_activity:   T x num_cells matrix
    %   Xstims:         cell array of stims, each of which is a Tx_ matrix
    %
    % OUTPUTS:
    %   none; throws error flag if parameters are not consistent
    %
    % CALLED BY:
    %   none
    
    % check filter dimensions
    assert(size(Xstims{subunit.x_target}, 2) == size(subunit.filt, 1), ...
        'Xstims dims inconsistent with stimulus filter')

    end % method
    
end
%% ********************  static methods ***********************************
methods (Static)
    
    function reg_lambdas = init_reg_lambdas()
    % reg_lambdas = StimSubunit.init_reg_lambdas()
    %
    % creates reg_lambdas struct and sets default values to 0
    %
    % INPUT:
    %   none
    % 
    % OUTPUT:
    %   reg_lambdas:    struct containing hyperparameter values for the
    %                   different reg penalties
    
    reg_lambdas.l2 = 0;     % L2 on filter coeffs
    reg_lambdas.d2t = 0;    % L2 on laplacian in temporal dim
    reg_lambdas.d2x = 0;    % L2 on laplacian in spatial dim
    reg_lambdas.d2xt = 0;   % L2 on spatio-temporal laplacian
    
    end % method
    
    
    function [filt, init_params] = set_init_filt_stat(stim_params, init_filt)
    % [filt,init_params] = StimSubunit.set_init_weights(stim_params, init_filt)
    % 
    % Initializes weights and sets init_params structure based on input. 
    % Called from the constructor (which is why its a static method) and 
    % from the non-static method set_init_weights
    %
    % INPUTS:
    %   stim_params:    struct array 
    %       dims:       defines dimensionality
    %                   of the (time-embedded) stimulus, in 
    %                   the form [num_lags, num_xpix, num_ypix].
    %       num_outputs num_cells ('individual') or 1 ('shared')
    %   init_weights:   either a string specifying type of random 
    %                   initialization for weights ('gaussian' or 
    %                   'uniform') or a weight vector of appropriate length
    % 
    % OUTPUTS:
    %   filt:           filter of appropriate size (defined by stim_params)
    %   init_params:    struct specifying init_weights and rng_state, if
    %                   applicable
    
    if ischar(init_filt)
        % randomly initialize filter
        init_params.stim_init_filt = lower(init_filt);
        init_params.rng_state = rng();
        % create filter
        s = 0.01;
        switch lower(init_filt)
            case 'gaussian'
                filt = s * randn(prod(stim_params.dims), ...
                                      stim_params.num_outputs);
            case 'uniform'
                filt = s * rand(prod(stim_params.dims), ...
                                      stim_params.num_outputs) - s/2;
            otherwise
                error('Invalid init_filt string')
        end
    elseif ismatrix(init_filt)
        % use 'init_filt' to initialize filter
        assert(isequal(size(init_filt), ...
            [prod(stim_params.dims), stim_params.num_outputs]), ...
            'init_filt matrix has improper size')
        filt = init_filt;
        % careful, this may tax memory with lots of subunits and large data
        init_params.stim_init_filt = filt; 
        init_params.rng_state = NaN;
    else
        warning('init_filt must be a string or a matrix')
    end
    
    end % method
    
    
	function stim_params = create_stim_params(dims, num_outputs, varargin)
    % stim_params = StimSubunit.create_stim_params(...
    %                                       stim_dims, num_outputs, kargs)
    %
    % Creates a struct containing stimulus parameters
    %
    % INPUTS:
    %   dims:           dimensionality of the (time-embedded) stimulus, in 
    %                   the form: [num_lags num_xpix num_ypix]. For 1 
    %                   spatial dimension use only num_xpix
    %   num_outputs:    number of outputs from the subunit; for 
    %                   'individual' subunit, this is equal to the number 
    %                   of cells; for 'shared' subunit, this is equal to 1.
    %     
    %   optional key-value pairs:
    %       'stim_dt', scalar
    %           time resolution (in ms) of stim matrix (used only for 
    %           plotting)
    %       'up_fac', scalar
    %           up-sampling factor of the stimulus from its raw form
    %       'tent_spacing', scalar
    %           spacing of tent-basis functions when using a tent-basis 
    %           representaiton of the stimulus (allows for the stimulus 
    %           filters to be represented at a lower time resolution than 
    %           other model components). 
    %       'boundary_conds', vector
    %           boundary conditions on each dimension (Inf is free, 0 is 
    %           tied to 0, and -1 is periodic)
    %       'split_pts', vector
    %           specifies an internal boundary as a 3-element vector: 
    %           [direction boundary_ind boundary_cond]
    %
    % OUTPUTS:
    %   stim_params: struct of stimulus parameters
    
    assert(nargin >= 2, ...
        'Not enough inputs')
    
    % Set defaults
    stim_dt = 1;                % default to unitless time
    up_fac = 1;                 % default no temporal up-sampling
    tent_spacing = [];          % default no tent-bases
    boundary_conds = [0 0 0];   % tied to 0 in all dims
    split_pts = [];             % no split points

    % Parse inputs
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'stim_dt'
                stim_dt = varargin{i+1};
            case 'up_fac'
                up_fac = varargin{i+1};
            case 'tent_spacing'
                tent_spacing = varargin{i+1};
            case 'boundary_conds'
                for j = 1:length(varargin{i+1})
                    assert(ismember(varargin{i+1}(j), [Inf, 0, -1]), ...
                        'Incorrect boundary condition specification')
                end
                boundary_conds = varargin{i+1};
            case 'split_pts'
                split_pts = varargin{i+1};
            otherwise
                error('Invalid input flag'); 
        end	
        i = i + 2;
    end

    % Make sure stim_dims input has form [num_lags num_xpix num_ypix] and
    % concatenate with 1's if necessary    
    while length(dims) < 3 
        % pad dims with 1s for book-keeping
        dims = cat(2, dims, 1);
    end

    % update matching boundary conditions
    while length(boundary_conds) < 3
        % assume free boundaries on spatial dims if not specified
        boundary_conds = cat(2, boundary_conds, 0); 
    end

    % set model fitting dt
    dt = stim_dt / up_fac; 

    % create struct to output
    stim_params = struct('dims', dims, 'dt', dt, 'up_fac', up_fac,...
                         'tent_spacing', tent_spacing, ...
                         'boundary_conds',boundary_conds,...
                         'split_pts',split_pts, ...
                         'num_outputs', num_outputs);

    end % method
	
    
    function Xmat = create_time_embedding(stim, stim_params)
    % Xmat = StimSubunit.create_time_embedding(stim)
    %
    % Takes a Txd stimulus matrix and creates a time-embedded matrix of
    % size Tx(d*num_lags), where num_lags is the desired number of time
    % lags specified in the stim_params struct. If stim is a 3d array the
    % spatial dimensions are folded into the 2nd dimension. Assumes
    % zero-padding. Note that Xmat is formatted so that adjacent time lags
    % are adjacent within a time-slice of Xmat. Thus Xmat(t,1:num_lags)
    % gives all the time lags of the first spatial pixel at time t.
    %
    % INPUTS:
    %   stim:   stimulus matrix; time must be in the first dim
    %
    % OUTPUTS:
    %   Xmat:   time-embedded stimulus matrix
    
    sz = size(stim);

    % if there are two spatial dims, fold them into one
    if length(sz) > 2
        stim = reshape(stim, sz(1), prod(sz(2:end)));
    end
    
    % no support for more than two spatial dims
    if length(sz) > 3
        warning('More than two spatial dimensions not supported; creating Xmat anyways...');
    end

    % check that the size of stim matches with the specified stim_params
    % structure
    [T, num_pix] = size(stim);
    if prod(stim_params.dims(2:end)) ~= num_pix
        error('Stimulus dimension mismatch');
    end

    % up-sample stimulus if required
    % find in NIMclass code
    
    % if using a tent-basis representation
    if ~isempty(stim_params.tent_spacing)
        tbspace = stim_params.tent_spacing;
        % create a tent-basis (triangle) filter
        tent_filter = [(1:tbspace) / tbspace ...
                        1 - (1:tbspace - 1) / tbspace] / tbspace;

        % apply to the stimulus
        filtered_stim = zeros(size(stim));
        for i = 1:length(tent_filter)
            filtered_stim = filtered_stim + ...
                StimSubunit.shift_mat_zpad(stim, i-tbspace, 1) * tent_filter(i);
        end

        stim = filtered_stim; 
        lag_spacing = tbspace;
    else
        lag_spacing = 1;
    end

    % for temporal only stimuli (this method can be faster if you're not 
    % using tent-basis rep
    if num_pix == 1
        Xmat = toeplitz(stim, [stim(1) zeros(1, stim_params.dims(1) - 1)]);
    else
        % otherwise loop over lags and manually shift the stim matrix
        Xmat = zeros(T, prod(stim_params.dims));
        for n = 1:stim_params.dims(1)
            Xmat(:,n-1+(1:stim_params.dims(1):(num_pix*stim_params.dims(1)))) = ...
                StimSubunit.shift_mat_zpad(stim, lag_spacing * (n-1), 1);
        end
    end
    
    end % method
    
    
    function tmat = split_tmat(tmat, split_pts)
    % tmat = StimSubunit.split_tmat(tmat, split_pts)
    % 
    % Creates new boundaries in a regularization matrix, specified by the
    % split_pts input
    %
    % INPUTS:
    %   tmat:       regularization matrix
    %   split_pts:  vector [direction boundary_ind boundary_cond]
    %
    % OUTPUTS:
    %   tmat:       updated regularization matrix
    
    split_loc = split_pts(2);
    split_bound = split_pts(3);

    % make the split on tmat
    tmat(split_loc,split_loc+1) = 0;
    tmat(split_loc+1,split_loc) = 0;

    if isinf(split_bound)
        % if splitting with free bounds
        tmat(:,[split_loc split_loc+1]) = 0;
    elseif split_bound == -1
        % if splitting with circ bounds
        tmat(split_loc,1) = 1; tmat(1,split_loc) = 1;
        tmat(split_loc+1,end) = 1; tmat(end,split_loc+1) = 1;
    end

    end % method
        
end
%% ********************* static hidden methods ****************************
methods (Static, Hidden)
    
    function Xshifted = shift_mat_zpad(X, shift, dim)
    % Xshifted = shift_mat_zpad(X, shift, <dim>)
    %
    % Takes a vector or matrix and shifts it along dimension dim by amount
    % shift using zero-padding. Positive shifts move the matrix right or 
    % down
    %
    % INPUTS:
    %   X:          matrix or vector to shift
    %   shift:      amount to shift by. positive shifts move the matrix right
    %               or down
    %   <dim>:      optional; dimension to shift along
    %
    % OUTPUTS:
    %   Xshifted:   shifted matrix or vector

    % default to appropriate dimension if X is one-dimensional
    if nargin < 3
        [a,~] = size(X);
        if a == 1
            dim = 2;
        else
            dim = 1;
        end
    end

    sz = size(X);
    if dim == 1
        if shift >= 0
            Xshifted = [zeros(shift,sz(2)); X(1:end-shift,:)];
        else
            Xshifted = [X(-shift+1:end,:); zeros(-shift,sz(2))];
        end
    elseif dim == 2
        if shift >= 0
            Xshifted = [zeros(sz(1),shift) X(:,1:end-shift)];
        else
            Xshifted = [X(:,-shift+1:end) zeros(sz(1),-shift)];
        end
    end
    
    end % method
    
end

end
