classdef Solver
    % Solver: class implementing the 
    
    properties
        dt          % [(N_t-1) x 1] scalar: time-step
        t           % [N_t x 1] array: time-stamps for observations
        
        u0 = 0      % [1 x 1] scalar: initial point for trajectory
        g = 9.8     % [1 x 1] scalar: acceleration due to gravity
        
        lam_eta     % [4 x 1] array: correlation parameters for numerical discrepancy covariance function
        V_eta       % [1 x 1] scalar: marginal variance for numerical discrepancy residual component
        
        % Data-generation domaind for eaoh of the individual parameters
        Dmn_gam = [0.1,2]
        Dmn_u0 = [0,0]
        Dmn_du0 = [5,15]
        Dmn_t = [0,2.5]
        Dmn_dt = [0.02,0.2]
        Dmn_u = [-10,10]
        
        % Bayes linear prior moments for the discrepancy model
        E_b
        Var_b
        % Bayes linear adjusted moments for the discrepancy model
        EF_b
        VarF_b 
        iVF
        iVF_D
        % copies of training data used for model fit
        phi_train
        F_train
        G_train
    end % properties
    
    properties(Dependent)
        N_t         % [1 x 1] scalar
    end % properties(Dependent)
    
    methods
        function obj = Solver(L_t,N_t,t)
            % Solver(.): constructor
            %
            % INPUTS:
            % - 
            % OUTPUTS:
            % - 
            
            % initialise time grid
            if nargin>2
                % use supplied t
                obj.t = t;
            else
                % setup evenly-spaced knots
                obj.t = linspace(0,L_t,N_t+1)';
            end
            % attach dt
            obj.dt = diff(obj.t);
        end % Solver
        
        function du = f(obj,u,gam,du0,t)
            % f(.): derivative function.
            %
            % INPUTS:
            % - u: [N_u x 1] array: initial point for trajectory
            % - gam: [1 x 1] scalar or [N_u x 1] array: air resistance parameter for
            %   trajectory.
            % - du0: [1 x 1] scalar or [N_u x 1] array: initial time-derivative parameter.
            % - it: [1 x 1] integer (1 <= it <= N_t): time-step of solution
            % OUTPUTS:
            % - du: [N_u x 1] array: derivatives evaluated at input points
            
            du = -gam.*(u-obj.u0) - obj.g.*t + du0;
        end % f
        
        function d2u = df(obj,u,gam,du0,t)
            % df(.): 2nd derivative function.
            %
            % INPUTS:
            % - u: [N_u x 1] array: initial point for trajectory
            % - gam: [1 x 1] scalar or [N_u x 1] array: air resistance parameter for
            %   trajectory.
            % - du0: [1 x 1] scalar or [N_u x 1] array: initial time-derivative parameter.
            % - it: [1 x 1] integer (1 <= it <= N_t): time-step of solution
            % OUTPUTS:
            % - du: [N_u x 1] array: derivatives evaluated at input points
            
            d2u = -gam.*obj.f(u,gam,du0,t) - obj.g;
        end % df
        
        function [u,uh,eta] = simulate_trajectory(obj,gam,du0,N_u,include_eta)
            % simulate_trajectory(.): simulate the trajectory for a given
            % set of parameters.
            %
            % INPUTS:
            % -
            % OUTPUTS:
            % - 
            
            % 
            if nargin<4
                N_u = numel(gam);
            end
            if nargin<5
                include_eta = false;
            end 
            % initialise
            u = nan(N_u,obj.N_t+1);
            uh = nan(N_u,obj.N_t);
            eta = nan(N_u,obj.N_t);
            u(:,1) = obj.u0;
            % loop over time stamps
            for iT = 1:obj.N_t
                uh(:,iT) = u(:,iT) + obj.dt(iT).*obj.f(u(:,iT),gam,du0,obj.t(iT));
                if include_eta
                    % inputs
                    phi.t = obj.t(iT).*ones(N_u,1);
                    phi.dt = obj.dt(iT).*ones(N_u,1);
                    phi.gam = gam;
                    phi.du0 = du0;
                    phi.u = u(:,iT);
                    % evaluate eta
                    [E_eta,Var_eta] = obj.bl_predict_eta(phi);
                    eta(:,iT) = E_eta + sqrt(diag(Var_eta)).*randn(N_u,1);
%                     eta(:,iT) = E_eta + 0.001.*randn(N_u,1);
                    % construct overall solution
                    u(:,iT+1) = uh(:,iT)+eta(:,iT);
                else
                    % solution with no sampled discrepancy
                    u(:,iT+1) = uh(:,iT);
                end 
            end % iT
        end % simulate_trajectory
        
        function u = true_trajectory(obj,gam,du0)
            % true_trajectory(.): evaluate the true trajectory
            %
            % INPUTS:
            % -
            % OUTPUTS:
            % - 
            
            u = (du0./gam + obj.g./(gam.^2)).*(1-exp(-gam.*(obj.t'))) - obj.g.*(obj.t')./gam;
        end % true_trajectory
        
        function obj = bl_update_eta(obj,F,phi)
            % bl_update_eta(.): carry out Bayes linear update of the model
            % parameters for the discrepancy model
            %
            % INPUTS:
            % - 
            % OUTPUTS:
            % -
            
            % attach data
            obj.F_train = F;
            obj.phi_train = phi;
            
            % Get basis
            G = obj.g_eta(phi);
            obj.G_train = G;
            
            % E[F]
            E_F = G*obj.E_b;
            % Var[F]
            Var_U = obj.cov_eta(phi,phi);
            Var_F = G*obj.Var_b*G' + Var_U;
            % Var[F]^{-1}
            obj.iVF = Var_F\eye(size(Var_F));
            obj.iVF_D = obj.iVF*(F-E_F);
            
            % E_F[b]
            Cov_b_F = obj.Var_b*G';
            obj.EF_b = obj.E_b + Cov_b_F*obj.iVF_D;
            % Var_F[b]
            obj.VarF_b = obj.Var_b - Cov_b_F*obj.iVF*Cov_b_F';
        end % bl_update_eta
        
        function [EF_eta,VarF_eta] = bl_predict_eta(obj,phi)
            % bl_predict_eta(.): predict 
            %
            % INPUTS:
            % -
            % OUTPUTS:
            % -
            
            % check that model update has already been perfrmed
            if isempty(obj.iVF_D)
                error('Fit discrepancy model before running bl_predict_eta')
            end
            
            % Cov[u(xp),u(x)]
            Cov_Up_U = obj.cov_eta(phi,obj.phi_train);
            % g(xp)
            Gp = obj.g_eta(phi);
            
            % E_F[b]*g(xp)
            EF_bG = Gp*obj.EF_b;
            % E_F[u(xp)]
            EF_U = Cov_Up_U*obj.iVF_D;
            
            % E_F[f(xp)]
            EF_eta = EF_bG + EF_U;
            
            % if requested: predictive variance
            if nargout>1
                % Var[u(xp)]
                Var_Up = obj.cov_eta(phi,phi);
                
                % g(xp)*Var_F[b]*g(xp)'
                GVarF_bG = Gp*obj.Var_b*Gp';
                % Var_F[u(xp)]
                VarF_Up = Var_Up - Cov_Up_U*obj.iVF*Cov_Up_U';
                % g(xp)*Cov_F[b,u(xp)]
                GCovF_b_Up = -(Gp*obj.Var_b*obj.G_train')*obj.iVF*Cov_Up_U';
                
                % Var_F[u(xp)]
                VarF_eta = GVarF_bG + VarF_Up + GCovF_b_Up + GCovF_b_Up';
            end
        end % bl_predict_eta
        
        function obj = fit_discrepancy_model(obj,N_Smp,N_Disc,N_Train,N_Test)
            % fit_discrepancy_model(.): get the moments for the discrepancy
            % model
            %
            % INPUTS:
            % - 
            % OUTPUTS:
            % - 
            
            % generate data fo initial prior setting
            [u_coarse_pr,u_acc_pr,phi_pr] = obj.gen_data(N_Train,50,N_Smp);
            % generate training data set
            [u_coarse_train,u_acc_train,phi_train] = obj.gen_data(N_Train,N_Disc,N_Smp);
            % get the test data set
            [u_coarse_test,u_acc_test,phi_test] = obj.gen_data(N_Test,N_Disc,N_Smp);
            
            %----------------------------------
            % Initialise the fixed parameters
            % G = basis
            G = obj.g_eta(phi_pr);
            
            % E[u^(acc)] - u^(coarse)
            E_eta = mean(u_acc_pr,2) - u_coarse_pr;
            % Var[u^(acc)]
            Var_eta = var(u_acc_pr,[],2);
            
            % Var[b]
            obj.Var_b = (G'*((1./Var_eta).*G))\eye(size(G,2));
            % E[b]
            obj.E_b = obj.Var_b*(G'*((1./Var_eta).*E_eta));
            
            % estimate of 
            obj.V_eta = var((E_eta - G*obj.E_b)./sqrt(phi_pr.dt));
            
            %----------------------------------
            % Run the full update
            
            % E[u^(acc)] - u^(coarse)
            E_eta_train = mean(u_acc_train,2) - u_coarse_train;
            % run fit function
            obj = obj.bl_update_eta(E_eta_train,phi_train);
            
            %----------------------------------
            % Check the performance
            
            % E_{F}[\eta] and Var_{F}[\eta] (Adjusted moments)
            [E_eta_test,Var_eta_test] = obj.bl_predict_eta(phi_test);
            % E[u^(acc)] - u^(coarse)
            eta_test = mean(u_acc_test,2) - u_coarse_test;
            % output standardised distance to prediction
            (eta_test-E_eta_test)./sqrt(diag(Var_eta_test) + var(u_acc_test,[],2))
        end % fit_discrepancy_model
        
        function C = cov_eta(obj,phi,phi_p)
            % cov_eta(.): covariance function for the numerical discrepancy
            %
            % INPUTS:
            % - 
            % OUTPUTS:
            % - 
            
            C = obj.V_eta.*bsxfun(@min,phi.dt,phi_p.dt').*...
                    exp(-(1/2).*(obj.lam_eta.t.*bsxfun(@minus,phi.t,phi_p.t').^2 +...
                                 obj.lam_eta.du0.*bsxfun(@minus,phi.du0,phi_p.du0').^2 +...
                                 obj.lam_eta.gam.*bsxfun(@minus,phi.gam,phi_p.gam').^2 +...
                                 obj.lam_eta.u.*bsxfun(@minus,phi.u,phi_p.u').^2));
        end % cov_ets
        
        function G = g_eta(obj,phi)
            % g_eta(.): basis function for the numerical discrepancy model
            %
            % INPUTS
            % - 
            % OUTPUTS:
            % - 
            
            % # inputs
            N_Data = size(phi.u,1);
            % evaluate basis
            G = [ones(N_Data,1),(1/2).*(phi.dt.^2).*obj.df(phi.u,phi.gam,phi.du0,phi.t)];
        end % g_eta
        
        function [u_coarse,u_acc,phi] = gen_data(obj,N_Data,N_Disc,N_Smp)
            % gen_data(.): generate data for the numerical discrepancy
            % model by running the solver on coarse and fine time steps.
            %
            % INPUTS:
            % - N_Data: [1 x 1] scalar: number of runs to generate
            % - N_Disc: [1 x 1] scalar: number of sub-segments in the fine
            %   discretization.
            % - N_Smp: [1 x 1] scalar: number of times to sample each
            %   trajectory.
            % OUTPUTS:
            % - u_coarse: [N_Data x N_Smp] array: coarse solver evaluations
            % - u_acc: [N_Data x N_Smp] array: 'accurate' (refined) solver
            %   evaluations.
            % - phi: [1 x 1] struct with fields {'t','dt','gam','du0','u'}:
            %   structure containing function input values.
            
            % list of variable names 
            Vrb_List = {'t','dt','gam','du0','u'};
            N_Vrb = numel(Vrb_List);
            
            % generate training set using a Latin hypercube
            LHD = lhsdesign(N_Data,N_Vrb);
            phi = [];
            for iVrb = 1:N_Vrb
                phi.(Vrb_List{iVrb}) = eval(sprintf('obj.Dmn_%s(1)',Vrb_List{iVrb})) +...
                                eval(sprintf('diff(obj.Dmn_%s)',Vrb_List{iVrb})).*LHD(:,iVrb);
            end % iVrb
            
            % generate initial u values by running solver
            % loop over data points
            for iDat = 1:N_Data
                % create solver obtect
                Slv_i = Solver(phi.t(iDat),N_Disc);
                % run solver
                u_i = Slv_i.simulate_trajectory(phi.gam(iDat),phi.du0(iDat),1,false);
                % extract final u
                phi.u(iDat) = u_i(end);
            end % iDat
            
            % generate coarse evaluations
            u_coarse = phi.u + phi.dt.*obj.f(phi.u,phi.gam,phi.du0,phi.t);
            
            % generate accurate evaluations
            u_acc = nan(N_Data,N_Smp);
            % loop over samples
            fprintf('Generating data for discrepancy model fit...\n')
            for iSmp = 1:N_Smp
                % iterate 
                u_iter = nan(N_Data,N_Disc+1);
                u_iter(:,1) = phi.u;
                t_iter = phi.t;
                dt_fine = phi.dt./N_Disc;
                for iDisc = 1:N_Disc
                    u_iter(:,iDisc+1) = u_iter(:,iDisc) +...
                            (dt_fine).*obj.f(u_iter(:,iDisc),phi.gam,phi.du0,t_iter) +...
                            (sqrt(0.01).*dt_fine).*randn(N_Data,1);
                    t_iter = t_iter + phi.dt./N_Disc;
                end % iDisc
                % store
                u_acc(:,iSmp) = u_iter(:,end);
                % count
                if rem(iSmp,10)==0
                    fprintf('#')
                    if rem(iSmp,100)==0
                        fprintf('\n')
                    end 
                end
            end % iSmp
            fprintf('Data generation complete.\n')
        end % gen_data
        
        %-----------------------------------------
        % get methods
        
        function N_t = get.N_t(obj)
            N_t = numel(obj.t)-1;
        end % get.N_t
    end % methods
end % Solver