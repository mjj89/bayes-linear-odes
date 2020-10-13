% 26/09/20: Re-do of the trajectory analysis example 

clear
close all
clc
restoredefaultpath
set(0,'defaultfigurewindowstyle','docked')

% add path to classes
addpath('../Classes')

%% Fit the numerical discrepancy model

% initialise
L_t = 3;
N_t = 50;
Slv = Solver(L_t,N_t);

% control
N_Smp = 100;
N_Disc = 100;
N_Train = 500;
N_Test = 100;

% set \lambda
Slv.lam_eta.t = 1./(3^2);
Slv.lam_eta.du0 = 1./(50^2);
Slv.lam_eta.gam = 1./(0.2^2);
Slv.lam_eta.u = 1./(50^2);

% run the model fit
Slv = Slv.fit_discrepancy_model(N_Smp,N_Disc,N_Train,N_Test);

%% Test the solver code

% specify params
N_u = 1000;
gam = 0.5+(0.15).*randn(N_u,1);
du0 = 30+(7.5).*randn(N_u,1);
sig = 1e-2;

% generate trajectories
[u,uh,eta] = Slv.simulate_trajectory(gam,du0,N_u,true);
u(:,1) = 1e-4.*randn(N_u,1);

% generate corresponding 
u_str = Slv.true_trajectory(gam,du0);

% plot
% figure(1)
% clf
% hold on
% J = jet(100);
% for iTrj = 1:100
%     plot(Slv.t,u(iTrj,:),'-','linewidth',1,'color',J(iTrj,:))
% end % iTrj
% plot(xlim,[0,0],'-k','linewidth',4)
% grid on
% box on
% xlabel('Time (s)')
% ylabel('u(t) (trajectory)')
% title('Plots of raw trajectory simulations')

%% Generate the DAG 

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
    if 0
        Pa{nd_ct} = [1;nd_ct-3;nd_ct-4];
    else
        if iT==1
            Pa{nd_ct} = [1;2];
        else
            Pa{nd_ct} = [1;nd_ct-3];
        end
    end
    % get covariance of parameters
    E{nd_ct} = mean(eta(:,iT));
    Cov{nd_ct,nd_ct} = mean((eta(:,iT)-E{nd_ct}).^2);
    % covariance with parents
    Cov{nd_ct,Pa{nd_ct}(1)} = (eta(:,iT)-mean(eta(:,iT)))'*(prm-mean(prm))./N_u;
    Cov{nd_ct,Pa{nd_ct}(2)} = (eta(:,iT)-mean(eta(:,iT)))'*(u(:,iT)-mean(u(:,iT)))./N_u;
    if 0
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

%% Convertt to junction tree

J = JunctionTree(G);

%% Observe a point as a test

% observe final point
id = [150;G.n;50];
% observe last sample
D{1} = {u_str(end,38)};
D{2} = {u_str(end,end)};
D{3} = {u_str(end,13)};
% observe final point
% id = G.n;
% % observe last sample
% D{1} = {u_str(end,end)};

% run the adjustment
J = J.SequentialAdjust(D,id);

%% Extract information about trajectory fom junction tree and plot

% init
% u
E_u_J = nan(Slv.N_t,1);
Var_u_J = nan(Slv.N_t,1);
ED_u_J = nan(Slv.N_t,1);
VarD_u_J = nan(Slv.N_t,1);
% uh
ED_uh_J = nan(Slv.N_t,1);
VarD_uh_J = nan(Slv.N_t,1);
% eta
ED_eta_J = nan(Slv.N_t,1);
VarD_eta_J = nan(Slv.N_t,1);
% loop over trajectory points
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
    ED_u_J(iT) = J.C(u_idx(1)).ED{strcmp(J.C(u_idx(1)).Lbl,u_nm)}{1};
    VarD_u_J(iT) = J.C(u_idx(1)).CovD{strcmp(J.C(u_idx(1)).Lbl,u_nm),strcmp(J.C(u_idx(1)).Lbl,u_nm)}{1,1};
    % uh
    ED_uh_J(iT) = J.C(uh_idx(1)).ED{strcmp(J.C(uh_idx(1)).Lbl,uh_nm)}{1};
    VarD_uh_J(iT) = J.C(u_idx(1)).CovD{strcmp(J.C(u_idx(1)).Lbl,uh_nm),strcmp(J.C(u_idx(1)).Lbl,uh_nm)}{1,1};
    % eta
    ED_eta_J(iT) = J.C(eta_idx(1)).ED{strcmp(J.C(eta_idx(1)).Lbl,eta_nm)}{1};
    VarD_eta_J(iT) = J.C(u_idx(1)).CovD{strcmp(J.C(u_idx(1)).Lbl,eta_nm),strcmp(J.C(u_idx(1)).Lbl,eta_nm)}{1,1};
end % iT

% plot the trajectory from this one update
figure(1)
clf
hold on
plot(Slv.t,u_str(end,:),'-k','linewidth',2)
% plot prior moments
plot(Slv.t(2:end),E_u_J,'-c','linewidth',2)
plot(Slv.t(2:end),E_u_J+2.*sqrt(Var_u_J),'--m','linewidth',2)
plot(Slv.t(2:end),E_u_J-2.*sqrt(Var_u_J),'--m','linewidth',2)
% plot adjusted moments
plot(Slv.t(2:end),ED_u_J,'-b','linewidth',2)
plot(Slv.t(2:end),ED_u_J+2.*sqrt(VarD_u_J),'--r','linewidth',2)
plot(Slv.t(2:end),ED_u_J-2.*sqrt(VarD_u_J),'--r','linewidth',2)
grid on
box on
xlabel('Time (s)')
ylabel('z(t) (m)')
