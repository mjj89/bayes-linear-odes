function obj = SequentialAdjust(obj,D,id)
% SequentialAdjust: adjust by 

% # data
nD = numel(id);

% if req.: initialise E_{D}[C_{i}] and Var_{D}[C_{i},C_{j}]
if ~isfield(obj.C(1),'ED')
    % loop over cliques
    for i = 1:obj.n
        obj.C(i).ED = obj.C(i).E;
        obj.C(i).CovD = obj.C(i).Cov;
    end
end

%% Sequentially adjust

% output
fprintf('Beginning sequential update...\n')

% loop over data
for iD = 1:nD
    % partially-adjusted object
    pob = obj;
    
    %--------------------------------------
    % Get data moments
    % E[D_{i}], Var[D_{i}]
    [Di,E_Di,Var_Di,clq,cel] = DataMoments(pob,D{iD},id(iD));
    % Var[D_{i}]^{-1}
%     iVar_Di = (Var_Di)\eye(size(Var_Di));
    iVar_Di = RobustInv(Var_Di);
    
    %--------------------------------------
    % Initialise propagation info
    
    % initial state
    trj.clq = clq;
    % initial boundary nodes
    trj.bnd = {id(iD)};
    % clique history
    trj.hstclq = clq;
    % covariance function
    trj.Cov{1} = eye(size(cell2mat(obj.C(clq).CovD{cel,cel})));
    
    %--------------------------------------
    % Propagate adjustment through graph
    
    % stopping indicator
    stop = false;
    % run
    while ~stop
        % adjust current cliques
        pob = AdjustCliques(obj,pob,trj,Di,E_Di,iVar_Di);
        % move outward to neighbouring cliques
        trj = UpdateState(obj,trj);
        
        % test for empty state
        if isempty(trj.clq)
            stop = true;
        end
    end
    
    % replace initial object
    obj = pob;
    
    % output
    fprintf('Update by data %g of %g done...\n',iD,nD)
end

% output
fprintf('Sequential update complete.\n')

return % SequentialAdjust

%% Get data moments
function [Di,E_Di,Var_Di,clq,cel] = DataMoments(obj,D,id)
% DataMoments: Get moments for an element of the data vector

% find node in cliques
all_clq = obj.CliqueMembership(id);
% just pick first in list
clq = all_clq(1);
% find node index within clique
cel = find(obj.C(clq).vid==id);

% init
Di = cell(obj.G.v(id).n,1);
E_Di = cell(obj.G.v(id).n,1);
Var_Di = cell(obj.G.v(id).n,obj.G.v(id).n);
for p = 1:obj.G.v(id).n
    % D_{i}
    Di{p} = D{p};
    % E[D_{i}]
    E_Di{p} = obj.C(clq).ED{cel}{p};
    for q = 1:obj.G.v(id).n
        % Var[D_{i}]
        Var_Di{p,q} = obj.C(clq).CovD{cel,cel}{p,q};
    end
end % p

% convert to matrices
Di = cell2mat(Di);
E_Di = cell2mat(E_Di);
Var_Di = cell2mat(Var_Di);

return

%% Propagate covariance
function newtrj = UpdateState(obj,trj)
% UpdateState: move to next set of cliques, and update covariance function

% find neighbours of current cliques
Ng = cell(numel(trj.clq),1);
for iC = 1:numel(trj.clq)
    % get all neighbours
    Ng{iC} = obj.C(trj.clq(iC)).ngb;
    % eliminate those already visited
    Ng{iC}(ismember(Ng{iC},trj.hstclq)) = [];
end

% delete empty cells
% eI = cellfun(@(c) isempty(c),Ng);
% Ng(eI) = [];

newCov = cell(numel(trj.clq),1);
newbnd = cell(numel(trj.clq),1);

% Find C_{i}\cap{}C_{j} and update state
for iC = 1:numel(trj.clq)
    for jC = 1:numel(Ng{iC})
        % C_{i}\cap{}C_{j}
        CinCj = obj.CliqueIntersection(trj.clq(iC),Ng{iC}(jC));
        % check for bad specification
        if isempty(CinCj)
            error('No intersection between neighbouring cliques!')
        end
        
        % Cov[b_{k},C_i\cap{}C_j]
        Cov_vk_CinCj = cell(numel(trj.bnd),numel(CinCj));
        for iV = 1:numel(trj.bnd{iC})
            for jV = 1:numel(CinCj)
                % find b_{k} in clique
                bI = obj.C(trj.clq(iC)).vid==trj.bnd{iC}(iV);
                % find C_{i}\cap{}C_{j} in clique
                iI = obj.C(trj.clq(iC)).vid==CinCj(jV);
                % get cov
                Cov_vk_CinCj{iV,jV} = cell2mat(obj.C(trj.clq(iC)).CovD{bI,iI});
            end % jV
        end % iV
        
        % Var[C_{i}\cap{}C_{j}]
        Var_CinCj = cell(numel(CinCj),numel(CinCj));
        for iV = 1:numel(CinCj)
            for jV = 1:numel(CinCj)
                % find v_{i} in clique
                iI = obj.C(trj.clq(iC)).vid==CinCj(iV);
                % find v_{j} in clique
                jI = obj.C(trj.clq(iC)).vid==CinCj(jV);
                % get cov
                Var_CinCj{iV,jV} = cell2mat(obj.C(trj.clq(iC)).CovD{iI,jI});
            end
        end
        
        % update covariance fun
        newCov{iC} = [newCov{iC};{trj.Cov{iC}*(cell2mat(Cov_vk_CinCj)*...
                          RobustInv(cell2mat(Var_CinCj)))}];
        % update clique boundary elements
        newbnd{iC} = [newbnd{iC};{CinCj}];
    end % jC
end % iC

% initialise trajectory
newtrj = trj;
% clear old elements
newtrj.Cov = [];
newtrj.bnd = [];
newtrj.clq = [];

% assign to trajectory
for iC = 1:numel(trj.clq)
    newtrj.Cov = [newtrj.Cov;newCov{iC}];
    newtrj.bnd = [newtrj.bnd;newbnd{iC}];
end

if ~isempty(Ng)
    for iC = 1:numel(trj.clq)
        % update list of visited cliques
        newtrj.hstclq = [newtrj.hstclq;Ng{iC}];
        % update current state
        newtrj.clq = [newtrj.clq;Ng{iC}];
    end
else
    % current state empty
    newtrj.clq = [];
end

return

%% Adjust current cliques
function pob = AdjustCliques(obj,pob,trj,Di,E_Di,iVar_Di)
% AdjustCliques: adjust cliques at current position

% loop over cliques
for iC = 1:numel(trj.clq)
    % loop over nodes
    for iV = 1:obj.C(trj.clq(iC)).n
        % compute Cov[C_{i}\cap{}C_{j},v_{k}] for {v_{k}} in C_{i}
        Cov_CinCj_vk = cell(numel(trj.bnd{iC}),1);
        for kV = 1:numel(trj.bnd{iC})
            i_int = obj.C(trj.clq(iC)).vid==trj.bnd{iC}(kV);
            Cov_CinCj_vk{kV} = cell2mat(obj.C(trj.clq(iC)).CovD{i_int,iV});
        end % jV
        
        % Cov[D_{i},v_{k}]
        Cov_Di_vk = trj.Cov{iC}*cell2mat(Cov_CinCj_vk);
        % E_{D_{i}}[v_{k}]
        EDi_vk = cell2mat(obj.C(trj.clq(iC)).ED{iV}) + Cov_Di_vk'*(iVar_Di*(Di-E_Di));
        % attach
        pob.C(trj.clq(iC)).ED{iV} = mat2cell(EDi_vk,...
                        obj.G.v(obj.C(trj.clq(iC)).vid(iV)).dm,1);
        % loop over nodes
        for jV = 1:iV
            % Cov[C_{i}\cap{}C_{j},v_{l}]
            Cov_CinCj_vl = cell(numel(trj.bnd{iC}),1);
            for kV = 1:numel(trj.bnd{iC})
                i_int = obj.C(trj.clq(iC)).vid==trj.bnd{iC}(kV);
                Cov_CinCj_vl{kV} = cell2mat(obj.C(trj.clq(iC)).CovD{i_int,jV});
            end
            Cov_Di_vl = trj.Cov{iC}*cell2mat(Cov_CinCj_vl);
            % Cov_{D_{i}}[v_{k},v_{l}]
            CovDi_vk_vl = cell2mat(obj.C(trj.clq(iC)).CovD{iV,jV}) -...
                    Cov_Di_vk'*(iVar_Di*Cov_Di_vl);
            % attach
            pob.C(trj.clq(iC)).CovD{iV,jV} = mat2cell(CovDi_vk_vl,...
                        obj.G.v(obj.C(trj.clq(iC)).vid(iV)).dm,...
                        obj.G.v(obj.C(trj.clq(iC)).vid(jV)).dm);
            pob.C(trj.clq(iC)).CovD{jV,iV} = mat2cell(CovDi_vk_vl',...
                        obj.G.v(obj.C(trj.clq(iC)).vid(jV)).dm,...
                        obj.G.v(obj.C(trj.clq(iC)).vid(iV)).dm);
        end % jV
    end % iV
end % iC

return

%% Robust inverse code
function iC = RobustInv(C)
% RobustInv: invert correlation matrix and std. deviations

% s -> std. dev. vector
s = sqrt(diag(C));
iS = diag(1./s);
% R -> corr matrix
R = iS*C*iS;
% R^{-1}
iR = (R+(1e-8).*eye(size(R)))\eye(size(R));

% re-combine
iC = iS*(iR*iS);

return