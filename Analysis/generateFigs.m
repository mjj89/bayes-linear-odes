% 16/10/2020: generate the figures for Bayes linear ODE paper

clear
close all
clc
restoredefaultpath
set(0,'defaultfigurewindowstyle','docked')

% add path to classes
addpath('../Classes')

% figure save flag
save_on = false;

if save_on
    % add path to figure save code
    addpath('C:\Users\Matthew.M.Jones2\LocalData\ExtremesCode\E_Code')
    % figure save directory
    PicDrc = 'C:\Users\Matthew.M.Jones2\OneDrive - Shell\Documents\Articles\ODEPaper\Paper_V5_CSDAReview\Figs_Updated';
end

% if required: set random number generator seed
% not necessary, but ensures consistency of figures produced.
rng(0)

%% Control parameters for the inference scheme

% initialise
L_t = 10;
N_t = 50;
t = [0;sort(L_t.*lhsdesign(N_t-1,1));L_t];
Slv = Solver(L_t,N_t,t);

% Numbers of samples etc. for fit
N_Smp = 100;
N_Disc = 500;
N_Train = 500;
N_Test = 100;

% Domains for model fits
Slv.Dmn_gam = [0.1,2];
Slv.Dmn_u0 = [0,0];
Slv.Dmn_du0 = [10,50];
Slv.Dmn_t = [0,8];
Slv.Dmn_dt = [0.02,0.3];
Slv.Dmn_u = [-10,10];

% set \lambda
Slv.lam_eta.t = 1./(3^2);
Slv.lam_eta.du0 = 1./(50^2);
Slv.lam_eta.gam = 1./(0.2^2);
Slv.lam_eta.u = 1./(50^2);

%% Fit the discrepancy model using the supplied ranges

Slv = Slv.fit_discrepancy_model(N_Smp,N_Disc,N_Train,N_Test);

%% Generate the solver trajectories for the prior moments

% prior moments for \gamma
E_gam = 0.55;
Var_gam = (0.2)^2;
% prior moments for du/dt(t_0)
E_du0 = 40;
Var_du0 = (10)^2;

% specify params
N_u = 2000;
gam = max(E_gam + sqrt(Var_gam).*randn(N_u,1),0.01);
du0 = max(E_du0 + sqrt(Var_du0).*randn(N_u,1),1);
sig = 1e-1;

% generate trajectories
[u,uh,eta] = Slv.simulate_trajectory(gam,du0,N_u,true);
% add on noise for initial state
u(:,1) = 1e-4.*randn(N_u,1);

%% Specify the DAG

% initialise
Pa = cell(4*Slv.N_t+2,1);
% E[.], Var[.] and Cov[.,.]
E = cell(4*Slv.N_t+2,1);
Cov = cell(4*Slv.N_t+2,4*Slv.N_t+2);
% node labels
Lbl = cell(4*Slv.N_t+2,1);
% original samples
Z = cell(4*Slv.N_t+2,1);

%-----------------------------
% [gam,du0]
% parents
Pa{1} = [];
% get covariance of parameters
prm = [gam,du0];
E{1} = mean(prm,1)';
Cov{1,1} = (prm'*prm)./N_u - E{1}*E{1}';
% label
Lbl{1} = 'xi';
% data
Z{1} = prm';

%-----------------------------
% u(t_0)
% parents
Pa{2} = [];
% get covariance of parameters
E{2} = mean(u(:,1));
Cov{2,2} = mean((u(:,1)-mean(u(:,1))).^2);
% label
Lbl{2} = 'u_0';
% data
Z{2} = u(:,1)';

% set parameter count
nd_ct = 3;

% loop over time steps
for iT = 1:Slv.N_t
    %-----------------------------
    % \hat{u}(t_k)
    % parents
    if iT==1
        Pa{nd_ct} = [1;2];
    else
        Pa{nd_ct} = [1;nd_ct-2];
    end
    % get covariance of parameters
    E{nd_ct} = mean(uh(:,iT));
    Cov{nd_ct,nd_ct} = mean((uh(:,iT)-E{nd_ct}).^2);
    % covariance with parents
    Cov{nd_ct,Pa{nd_ct}(1)} = (uh(:,iT)-mean(uh(:,iT)))'*(prm-mean(prm))./N_u;
    Cov{nd_ct,Pa{nd_ct}(2)} = (uh(:,iT)-mean(uh(:,iT)))'*(u(:,iT)-mean(u(:,iT)))./N_u;
    % label
    Lbl{nd_ct} = sprintf('uh_%g',iT);
    % data
    Z{nd_ct} = uh(:,iT)';
    
    % set parameter count
    nd_ct = nd_ct+1;
    
    %-----------------------------
    % \eta(t_k)
    % parents
    if iT==1
        Pa{nd_ct} = [1;2];
    else
        Pa{nd_ct} = [1;nd_ct-3;nd_ct-4];
    end
    % get covariance of parameters
    E{nd_ct} = mean(eta(:,iT));
    Cov{nd_ct,nd_ct} = mean((eta(:,iT)-E{nd_ct}).^2);
    % covariance with parents
    Cov{nd_ct,Pa{nd_ct}(1)} = (eta(:,iT)-mean(eta(:,iT)))'*(prm-mean(prm))./N_u;
    Cov{nd_ct,Pa{nd_ct}(2)} = (eta(:,iT)-mean(eta(:,iT)))'*(u(:,iT)-mean(u(:,iT)))./N_u;
    if iT>1
        Cov{nd_ct,Pa{nd_ct}(3)} = (eta(:,iT)-mean(eta(:,iT)))'*(eta(:,iT-1)-mean(eta(:,iT-1)))./N_u;
    end
    % label
    Lbl{nd_ct} = sprintf('eta_%g',iT);
    % data
    Z{nd_ct} = eta(:,iT)';
    
    % set parameter count
    nd_ct = nd_ct+1;
    
    %-----------------------------
    % u(t_k)
    % parents
    Pa{nd_ct} = [nd_ct-2;nd_ct-1];
    % get covariance of parameters
    E{nd_ct} = mean(u(:,iT+1));
    Cov{nd_ct,nd_ct} = mean((u(:,iT+1)-E{nd_ct}).^2);
    % covariance with parents
    Cov{nd_ct,Pa{nd_ct}(1)} = (u(:,iT+1)-mean(u(:,iT+1)))'*(uh(:,iT)-mean(uh(:,iT)))./N_u;
    Cov{nd_ct,Pa{nd_ct}(2)} = (u(:,iT+1)-mean(u(:,iT+1)))'*(eta(:,iT)-mean(eta(:,iT)))./N_u;
    % label
    Lbl{nd_ct} = sprintf('u_%g',iT);
    % data
    Z{nd_ct} = u(:,iT+1)';
    
    % set parameter count
    nd_ct = nd_ct+1;
    
    %-----------------------------
    % z(t_k)
    % parents
    Pa{nd_ct} = nd_ct-1;
    % get covariance of parameters
    z_k = u(:,iT+1) + (sig).*randn(N_u,1);
    E{nd_ct} = mean(z_k);
    Cov{nd_ct,nd_ct} = mean((z_k-E{nd_ct}).^2);
    % covariance with parents
    Cov{nd_ct,Pa{nd_ct}} = (z_k-mean(z_k))'*(u(:,iT+1)-mean(u(:,iT+1)))./N_u;
    % label
    Lbl{nd_ct} = sprintf('z_%g',iT);
    % data
    Z{nd_ct} = z_k';
    
    % set parameter count
    nd_ct = nd_ct+1;
end % iT

% convert everything to cells
for i = 1:numel(Pa)
    E{i} = num2cell(E{i});
    for j = 1:numel(Pa)
        if ~isempty(Cov{i,j})
            Cov{i,j} = num2cell(Cov{i,j});
        end
    end
end % i

% use these to initialise DAG
G = DAG(Pa,E,Cov,Lbl,Z);

%% Convert the DAG to a junction tree

J = JunctionTree(G);

%% Observe trajectories for a number of different parameter specifications
% Generaate the corresponding adjusted moments

% parameter values for real trajectories
% \gamma
gam_case = [0.8;0.7;0.6;0.4;0.3];
% du_0 
du0_case = [25;30;40;45;50];

% set observation locations for 
idx_obs = [4,10,17;
           6,12,20;
           8,15,23;
           10,17,28;
           11,25,33];
N_Obs = size(idx_obs,2);
% # cases
N_Case = numel(gam_case);
% real solutions
u_real = nan(Slv.N_t+1,N_Case);

% initialise storage for moments
% u
ED_u_J = nan(Slv.N_t,N_Case);
VarD_u_J = nan(Slv.N_t,N_Case);
% uh
ED_uh_J = nan(Slv.N_t,N_Case);
VarD_uh_J = nan(Slv.N_t,N_Case);
% eta
ED_eta_J = nan(Slv.N_t,N_Case);
VarD_eta_J = nan(Slv.N_t,N_Case);
% xi
ED_xi_J = nan(2,N_Case);
VarD_xi_J = nan(2,2,N_Case);

% loop over cases
for iCase = 1:N_Case
    %----------------------------------------
    % Generate the trajectory and perform the adjustment
    % generate the real trajectory
    u_real(:,iCase) = Slv.true_trajectory(gam_case(iCase),du0_case(iCase));
    % copy junction tree
    J_case = J;
    % format the input data
    % find the node numbers of the observed points on the DAG
    idx_G = nan(N_Obs,1);
    for iOb = 1:N_Obs
        z_name = sprintf('z_%g',idx_obs(iCase,iOb));
        idx_G(iOb) = find(cellfun(@(x)any(strcmp(x,z_name)),{G.v.Lbl}','UniformOutput',true));
    end % iOb
    % setup input data structure
    D_case = num2cell(u_real(idx_obs(iCase,:)+1,iCase));
    for iOb = 1:N_Obs
        D_case{iOb} = {D_case{iOb}};
    end % iOb
    % perform the adjustment
    J_case = J_case.SequentialAdjust(D_case,idx_G);
    
    %----------------------------------------
    % Extract the adjusted moments
    % extract the parameter moments
    ED_xi_J(:,iCase) = cell2mat(J_case.C(1).ED{1});
    VarD_xi_J(:,:,iCase) = cell2mat(J_case.C(1).CovD{1,1});
    % extract trajectory components
    for iT = 1:Slv.N_t
        % get names
        u_nm = sprintf('u_%g',iT);
        uh_nm = sprintf('uh_%g',iT);
        eta_nm = sprintf('eta_%g',iT);
        % identify cliques containing variables
        u_idx = find(cellfun(@(x)any(strcmp(x,u_nm)),{J_case.C.Lbl}','UniformOutput',true));
        uh_idx = find(cellfun(@(x)any(strcmp(x,uh_nm)),{J_case.C.Lbl}','UniformOutput',true));
        eta_idx = find(cellfun(@(x)any(strcmp(x,eta_nm)),{J_case.C.Lbl}','UniformOutput',true));
        % extract info from the cliques
        % u
        ED_u_J(iT,iCase) = J_case.C(u_idx(1)).ED{strcmp(J_case.C(u_idx(1)).Lbl,u_nm)}{1};
        VarD_u_J(iT,iCase) = J_case.C(u_idx(1)).CovD{strcmp(J_case.C(u_idx(1)).Lbl,u_nm),strcmp(J_case.C(u_idx(1)).Lbl,u_nm)}{1,1};
        % uh
        ED_uh_J(iT,iCase) = J_case.C(uh_idx(1)).ED{strcmp(J_case.C(uh_idx(1)).Lbl,uh_nm)}{1};
        VarD_uh_J(iT,iCase) = J_case.C(uh_idx(1)).CovD{strcmp(J_case.C(uh_idx(1)).Lbl,uh_nm),strcmp(J_case.C(uh_idx(1)).Lbl,uh_nm)}{1,1};
        % eta
        ED_eta_J(iT,iCase) = J_case.C(eta_idx(1)).ED{strcmp(J_case.C(eta_idx(1)).Lbl,eta_nm)}{1};
        VarD_eta_J(iT,iCase) = J_case.C(eta_idx(1)).CovD{strcmp(J_case.C(eta_idx(1)).Lbl,eta_nm),strcmp(J_case.C(eta_idx(1)).Lbl,eta_nm)}{1,1};
    end % iT
end % iCase

% extract the prior moments
% parameter moments
E_xi_J = cell2mat(J.C(1).E{1});
Var_xi_J = cell2mat(J.C(1).Cov{1,1});
% u
E_u_J = nan(Slv.N_t,1);
Var_u_J = nan(Slv.N_t,1);
% uh
E_uh_J = nan(Slv.N_t,1);
Var_uh_J = nan(Slv.N_t,1);
% eta
E_eta_J = nan(Slv.N_t,1);
Var_eta_J = nan(Slv.N_t,1);
% extract trajectory components
for iT = 1:Slv.N_t
    % get names
    u_nm = sprintf('u_%g',iT);
    uh_nm = sprintf('uh_%g',iT);
    eta_nm = sprintf('eta_%g',iT);
    % identify cliques containing variables
    u_idx = find(cellfun(@(x)any(strcmp(x,u_nm)),{J.C.Lbl}','UniformOutput',true));
    uh_idx = find(cellfun(@(x)any(strcmp(x,uh_nm)),{J.C.Lbl}','UniformOutput',true));
    eta_idx = find(cellfun(@(x)any(strcmp(x,eta_nm)),{J.C.Lbl}','UniformOutput',true));
    % extract info from the cliques
    % u
    E_u_J(iT) = J.C(u_idx(1)).E{strcmp(J.C(u_idx(1)).Lbl,u_nm)}{1};
    Var_u_J(iT) = J.C(u_idx(1)).Cov{strcmp(J.C(u_idx(1)).Lbl,u_nm),strcmp(J.C(u_idx(1)).Lbl,u_nm)}{1,1};
    % uh
    E_uh_J(iT) = J.C(uh_idx(1)).E{strcmp(J.C(uh_idx(1)).Lbl,uh_nm)}{1};
    Var_uh_J(iT) = J.C(uh_idx(1)).Cov{strcmp(J.C(uh_idx(1)).Lbl,uh_nm),strcmp(J.C(uh_idx(1)).Lbl,uh_nm)}{1,1};
    % eta
    E_eta_J(iT) = J.C(eta_idx(1)).E{strcmp(J.C(eta_idx(1)).Lbl,eta_nm)}{1};
    Var_eta_J(iT) = J.C(eta_idx(1)).Cov{strcmp(J.C(eta_idx(1)).Lbl,eta_nm),strcmp(J.C(eta_idx(1)).Lbl,eta_nm)}{1,1};
end % iT

% attach initial states
% prior cases
E_u_J = [0;E_u_J];
Var_u_J = [0;Var_u_J];
E_uh_J = [0;E_uh_J];
Var_uh_J = [0;Var_uh_J];
E_eta_J = [0;E_eta_J];
Var_eta_J = [0;Var_eta_J];
% adjusted cases
ED_u_J = [zeros(1,N_Case);ED_u_J];
VarD_u_J = [zeros(1,N_Case);VarD_u_J];
ED_uh_J = [zeros(1,N_Case);ED_uh_J];
VarD_uh_J = [zeros(1,N_Case);VarD_uh_J];
ED_eta_J = [zeros(1,N_Case);ED_eta_J];
VarD_eta_J = [zeros(1,N_Case);VarD_eta_J];

%% Plots of the results

% Figure 1: plot the trajectory results
figure(1)
clf
hold on
L = lines(N_Case);
Lgd = cell(N_Case,1);
% plot prior
plot(Slv.t,E_u_J,'-c','linewidth',2,'handlevisibility','off')
plot(Slv.t,E_u_J+3.*sqrt(Var_u_J),'--m','linewidth',2,'handlevisibility','off')
plot(Slv.t,E_u_J-3.*sqrt(Var_u_J),'--m','linewidth',2,'handlevisibility','off')
% plot cases
for iCase = 1:N_Case
    % plot real trajectory
    plot(Slv.t,u_real(:,iCase),'-k','linewidth',2,'handlevisibility','off')
    % plot adjusted moments
    plot(Slv.t,ED_u_J(:,iCase),'-','linewidth',1.5,'color',L(iCase,:))
    plot(Slv.t,ED_u_J(:,iCase)+3.*sqrt(VarD_u_J(:,iCase)),'--','linewidth',1.5,'color',L(iCase,:),'handlevisibility','off')
    plot(Slv.t,ED_u_J(:,iCase)-3.*sqrt(VarD_u_J(:,iCase)),'--','linewidth',1.5,'color',L(iCase,:),'handlevisibility','off')
    % plot observation points
    plot(Slv.t(idx_obs(iCase,:)),ED_u_J(idx_obs(iCase,:),iCase),'.k','markersize',40,'handlevisibility','off')
    % legend
    Lgd{iCase} = sprintf('gam = %.2f, u_t(t_0) = %g',gam_case(iCase),du0_case(iCase));
end % iCase
% format
ylim([-25,100])
xlim([0,10])
plot(xlim,[0,0],'-k','linewidth',3)
grid on
box on
legend(Lgd,'location','best')
xlabel('Time (s)')
ylabel('z(t) (m)')
set(gca,'fontsize',14)
% save
if save_on
    savePics(fullfile(PicDrc,'Fig7181'),9,6)
end

% Figure 2: plot the 
figure(2)
clf
hold on
% knots for the ellipse plots
nE = 200;
tt = linspace(0,2*pi,nE);
tx = cos(tt); ty = sin(tt); tX = [tx;ty];
% plot prior
% eigenvalues of covariance matrix
[Ev,ev] = eig(Var_xi_J);
% plot mean
plot(E_xi_J(1),E_xi_J(2),'.m','markersize',30)
% plot contours
for iS = 1:3
    % ellipse
    Elp = bsxfun(@plus,E_xi_J,Ev*sqrt(ev)*(iS.*tX));
    % plot
    plot(Elp(1,:),Elp(2,:),'--m','linewidth',2)
end % iS
% plot cases
for iCase = 1:N_Case
    % plot mean
    plot(ED_xi_J(1,iCase),ED_xi_J(2,iCase),'.','markersize',30,'color',L(iCase,:))
    % eigenvalues of covariance matrix
    [Ev,ev] = eig(VarD_xi_J(:,:,iCase));
    % plot contours
    for iS = 1:3
        % ellipse
        Elp = bsxfun(@plus,ED_xi_J(:,iCase),Ev*sqrt(ev)*(iS.*tX));
        % plot
        plot(Elp(1,:),Elp(2,:),'--','linewidth',2,'color',L(iCase,:))
    end % iS
    % plot true value
    plot(gam_case(iCase),du0_case(iCase),'.k','markersize',30)
end % iCase
% format
xlim([-0.1,1.2])
ylim([5,75])
grid on
box on
xlabel('\gamma{} (s^{-1})')
ylabel('du/dt(t_0) (m/s)')
set(gca,'fontsize',14)
% save
if save_on
    savePics(fullfile(PicDrc,'Fig7183'),9,6)
end