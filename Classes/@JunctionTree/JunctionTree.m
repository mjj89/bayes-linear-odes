classdef JunctionTree
    % JunctionTree: converts supplied DAG into a junction tree, and carries
    % out Bayes linear analysis on the junction tree.
    %
    % Procedure for conversion of a DAG to a junction tree described in
    % Appendix B.3 of the manuscript.
    
    %% Properties
    properties
        G       % [1 x 1] DAG object: original directed acyclic graph specification
        C       % [n x 1] struct: clique objects
        n       % [1 x 1] scalar: total # cliques in the model
    end % properties
    
    methods
        %% Constructor
        function obj = JunctionTree(G)
            % JunctionTree(.): construct a Junction tree from a DAG object
            %
            % INPUTS:
            % - G: [1 x 1] DAG object: original DAG specification to be
            %   converted to a Junction tree.
            % OUTPUTS:
            % - obj: [1 x 1] JunctionTree object: corresponding to the
            % original DAG specification.
            
            % attach DAG
            obj.G = G;
            
            % create junction tree from DAG
            obj = obj.makeJTfromDAG();
        end % JunctionTree
        
        %% Identify clique membership
        function cid = CliqueMembership(obj,v_id)
            % ClqMembership(.): find clique memberships of node v_{i}
            % 
            % INPUTS:
            % - v_id: [1 x 1] integer, or string: node specification
            %   (either node label, or node index)
            % OUTPUTS:
            % - cid: [N_Mem x 1] integer array: clique memberships of the
            %   specified node (clique indices).
            
            % if label used -> convert input type
            if ischar(v_id)
                v_id = strcmp(v_id,{obj.G.v(:).Lbl});
            end
            
            % loop over cliques
            for i = 1:obj.n
                cid(i) = ismember(v_id,obj.C(i).vid);
            end
            cid = find(cid);
        end % CliqueMembership
        
        %% Establish clique intersection
        function cis = CliqueIntersection(obj,i,j)
            % CliqueIntersection(.): find intersection between cliques i 
            % and j.
            %
            % INPUTS:
            % - i: [1 x 1] integer: first clique index.
            % - j: [1 x 1] integer: second clique index.
            % OUTPUTS:
            % - cis: [N_Int x 1] integer array: indices of intersection
            %   nodes in original DAG.
            
            % find intersection
            cis = intersect(obj.C(i).vid,obj.C(j).vid);
        end % CliqueIntersection
        
        %% Create junction tree from original DAG
        function obj = makeJTfromDAG(obj)
            % makeJTfromDAG(.): create a junction tree from a DAG 
            %
            % INPUTS:
            % - obj: [1 x 1] JunctionTree.
            % OUTPUTS:
            % - obj: updated with clique information.
            
            % get cliques from graph
            [CLst,A] = obj.G.Cliques();
            % get full graph covariance
%             Cov = obj.G.Covariance(A,3);
            Cov = obj.G.CovarianceNumerical(A,3);
            % # cliques
            obj.n = numel(CLst);
            
            % construct cliques C_{i}
            for i = 1:obj.n
                % node list
                obj.C(i,1).vid = CLst{i};
                % # component nodes
                obj.C(i,1).n = numel(CLst{i});
                % labels
                obj.C(i,1).Lbl = cell(obj.C(i,1).n,1);
                % init
                obj.C(i,1).E = cell(obj.C(i,1).n,1);
                obj.C(i,1).Cov = cell(obj.C(i,1).n,obj.C(i,1).n);
                % populate
                for j = 1:obj.C(i,1).n
                    % Label
                    obj.C(i,1).Lbl{j} = obj.G.v(obj.C(i,1).vid(j)).Lbl;
                    % E[C_{ij}]
                    obj.C(i,1).E{j} = obj.G.v(obj.C(i,1).vid(j)).E;
                    for k = 1:obj.C(i,1).n
                        % Cov[C_{ij},C_{ik}]
                        obj.C(i,1).Cov{j,k} =...
                            Cov{obj.C(i,1).vid(j),obj.C(i,1).vid(k)};
                    end % k
                end % j
            end % i
            
            % do max. cardinality search
            card = obj.G.MaxCardinality(A);
            
            % sort the cliques
            obj = obj.OrderCliques(card);
            
            % get clique ordering & neighbourhood structure
            obj = obj.Neighbourhood();
        end % ClqEdges
        
        %% Order the cliques
        function obj = OrderCliques(obj,card)
            % OrderCliques(.): order the cliques in terms of the highest
            % numbered member node.
            % 
            % INPUTS:
            % - card: [obj.G.n x 1] array: labelling calculated from max.
            %   cardinality search.
            % OUTPUTS:
            % - obj: [1 x 1] JunctionTree: with cliques sorted.
            
            % loop over cliques
            imx = nan(obj.n,1);
            for i = 1:obj.n
                % find max
                imx(i) = max(card(obj.C(i).vid));
            end
            % unique max els
            uq_imx = unique(imx);
            
            % sort in terms of max. cardinalty ord
            % where there are multiple cliques with the same max, sort in
            % terms of the other nodes
            % init
            sI = [];
            % loop over 
            for i = 1:numel(uq_imx)
                % find cliques with this max. crd.
                cI = imx==uq_imx(i);
                % cases
                if sum(cI)==1
                    % assign
                    sI = [sI;find(cI)];
                else
                    % multiple
                    % get vertex indices
                    c_idx = find(cI);
                    c_vid = cell(numel(c_idx),1);
                    for j = 1:numel(c_idx)
                        c_vid{j} = card(obj.C(c_idx(j)).vid);
                    end
                    % if necessary, pad with zeros
                    nv_max = max(cellfun(@(x)numel(x),c_vid));
                    Vmat = nan(numel(c_vid),nv_max);
                    for j = 1:numel(c_vid)
                        Vmat(j,:) = [sort(c_vid{j},'descend');...
                                     zeros(nv_max-numel(c_vid{j}),1)]';
                    end
                    % sort rows
                    [sVmat,tsI] = sortrows(Vmat,'ascend');
                    % assign
                    sI = [sI;c_idx(tsI)];
                end
            end
            
            % sort
            obj.C = obj.C(sI);
            
            % if there are multiple cliques with the same highest-numbered
            % node, order in terms of size
        end % OrderCliques
        
        %% Get neighbourhood structure
        function obj = Neighbourhood(obj)
            % Neighbourhood: find neighbourhood structure of the juncion
            % tree
            %
            % INPUTS:
            % - obj: [1 x 1] JunctionTree.
            % OUTPUTS:
            % - obj: updated with neighbourhood information for the
            %   cliques.
            
            % output
            fprintf('Finding neighbourhood structure...\n')
            
            % initialise neighbourhood fields
            for i = 1:obj.n
                obj.C(i).ngb = [];
            end % i
            
            % loop over cliques
            for i = 2:obj.n
                % find nodes which intersect with lower-numbered cliques
                v_int = cellfun(@(x)intersect(obj.C(i).vid,x),...
                                    {obj.C(1:(i-1)).vid}','UniformOutput',false);
                uq_int = unique(cell2mat(v_int));
                % find lower-numbered cliques which completely contains
                % this intersection
                link = find(cellfun(@(x)isempty(setdiff(uq_int,x)),...
                            cellfun(@(x)intersect(uq_int,x),v_int,'UniformOutput',false)));
                % link to the highest-numbered clique
                obj.C(i).ngb = [obj.C(i).ngb;link(end)];
                obj.C(link(end)).ngb = [obj.C(link(end)).ngb;i];
                % count
                if rem(i,10)==0
                    fprintf('#')
                    if rem(i,100)==0
                        fprintf('\n')
                    end
                end
            end % i
            
            % output
            fprintf('Neighbourhood structure identified.\n')
        end % Neighbourhood
        
        %% Sequential adjust
        obj = SequentialAdjust(obj,D,Var_e,id)
    end % methods
end % JunctionTree

