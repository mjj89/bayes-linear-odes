function obj = SequentialAdjust(obj,D,id)
% SequentialAdjust(.): adjust the moments on the junction tree given the
% input data.
% Accepts as input a cell array (D) of data for the adjustment, along with
% a vector (id) of the node indices of the data points (into the original
% DAG).
%
% Adjustment of moments on a Junction tree described in Appendix B.3 of the
% article, illustrated in a simple example in Appendix B.4, and described
% for a trajectory ODE example in Section 5.2.
%
% INPUTS:
% - obj: [1 x 1] JunctionTree object: junction tree model to be adjusted.
% - D: [N_D x 1] cell array: data for the adjustment (each supplied data
%   point structured in cells according to the equivalent node in the
%   graph).
% - id: [N_D x 1] index array: DAG node indices of the supplied data
%   points.
% OUTPUTS:
% - obj: JunctionTree, updated with the adjusted moments for the cliques.

% # data
nD = numel(id);

% if required: initialise E_{D}[C_{i}] and Var_{D}[C_{i},C_{j}]
% if E_{D}[C_{i}] and Var_{D}[C_{i},C_{j}] already exist, then we continue
% the adjustment, using the current state for the prior moments.
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
% DataMoments(.): initial function for identifying the moments of the
% observed data points from the model.
% 
% INPUTS:
% - obj: [1 x 1] JunctionTree object: junction tree model to be adjusted.
% - D: [obj.G.v(id).n x 1] cell array: observed data for the specified
%   node.
% - id: [1 x 1] index array: DAG node indices of the supplied data
%   point.
% OUTPUTS:
% - Di: [sum(obj.G.v(id).dm) x 1] array: data converted into array.
% - E_Di: [sum(obj.G.v(id).dm) x 1] array: prior expectation for the
%   supplied data.
% - Var_Di: [sum(obj.G.v(id).dm) x sum(obj.G.v(id).dm)] array: prior
%   covariance matrix for the supplied data.
% - clq: [1 x 1] integer: single clique of which DAG node id is a member. 
% - cel: [1 x 1] integer: index of the observed node within the clique clq.

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

return % DataMoments

%% Propagate covariance
function newtrj = UpdateState(obj,trj)
% UpdateState: move to next set of cliques, and update covariance function
% correspondingly.
%
% INPUTS:
% - obj: [1 x 1] JunctionTree object.
% - trj: [1 x 1] struct: information about the current state of the
%   adjustment pass (current clique location, covariance of separating
%   subset with observed data, etc.)
% OUTPUTS:
% - newtrj: [1 x 1] struct: trajectory object, moved to the next set of
%   un-visited neighbouring cliques.

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
            end % jV
        end % iV
        
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
end % iC

if ~isempty(Ng)
    for iC = 1:numel(trj.clq)
        % update list of visited cliques
        newtrj.hstclq = [newtrj.hstclq;Ng{iC}];
        % update current state
        newtrj.clq = [newtrj.clq;Ng{iC}];
    end % iC
else
    % current state empty
    newtrj.clq = [];
end

return % UpdateState

%% Adjust current cliques
function pob = AdjustCliques(obj,pob,trj,Di,E_Di,iVar_Di)
% AdjustCliques: adjust the cliques at the current state.
%
% INPUTS:
% - obj: [1 x 1] JunctionTree: fixed, before update by latest data point:
%   gives a fixed copy of the prior moments.
% - pob: [1 x 1] JunctionTree: copy of junction tree, where clique moments
%   are iteratively updated.
% - trj: [1 x 1] struct: current trajectory state.
% - Di: [sum(obj.G.v(id).dm) x 1] array: observed data for adjustment.
% - E_Di: [sum(obj.G.v(id).dm) x 1] array: prior data expectation.
% - iVar_Di: [sum(obj.G.v(id).dm) x sum(obj.G.v(id).dm)] array: inverse
%   prior covariance matrix for the data.
% OUTPUTS:
% - pob: [1 x 1] JunctionTree: with clique moments for the current state
%   updated.

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

return % AdjustCliques

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