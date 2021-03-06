classdef DAG
    % DAG: properties and methods for a Bayes linear DAG
    %
    % Bayes linear directed graphical models introduced in Appendix B.1 of
    % the article.
    % Graphical model for the general ODE problem introduced in Section 3.2
    % of the article.
    
    %% Properties
    properties
        n       % [1 x 1] scalar: number of vertices in 
        v       % [n x 1] struct: individual vertices (properties attached)
        
        % Sub-fields of v:
        %   n       [1 x 1] scalar: number of variables associated with
        %           vertex.
        %   dm      [n x 1] integer array: dimension of each variable
        %           associated with the vertex.
        %   E       [n x 1] cell array (containing [dm(i) x 1] arrays):
        %           variable expectations for this vertex.
        %   Var     [n x n] cell array (containing [dm(i) x dm(i)] arrays):
        %           variable covariance matrix assocaited with this vertex.
        %   Cov     [1 x N_Pa] cell array (containing [n x n(Pa_i)] cell
        %           arrays): covariances of variables associated with this
        %           vertex with variables in its parent vertices.
        %   Lbl     String: vertex label
        %   Z       [n x 1] cell array (containing [dm(i) x N_Smp] arrays):
        %           samples of variables associated with this vertex.
    end % properties
    
    methods
        %% Constructor
        function obj = DAG(Pa,E,Cov,Lbl,Z)
            % DAG: construct a DAG instance.
            %
            % Bayes linear directed graphical models introduced in Appendix B.1 of
            % the article.
            % Graphical model for the general ODE problem introduced in Section 3.2
            % of the article.
            %
            % INPUTS:
            % - Pa: [n x 1] cell array: list of parents for each vertex in
            %   the DAG.
            % - E: [n x 1] cell array: list of variable expectatoions for
            %   each vertex.
            % - Cov: {n x n} cell array: array of vertex covariances.
            % - Lbl: [n x 1] cell array (optional): list of vertex labels.
            % - Z: [n x 1] cell array (optional): list of samples for 
            %   variables corresponding to each vertex. 
            % OUTPUTS:
            % - obj: [1 x 1] DAG object.
            
            % # nodes
            obj.n = numel(Pa);
            % vertices
            for i = 1:obj.n
                % Pa(v_{i})
                obj.v(i,1).Pa = Pa{i};
                if nargin>1
                    % n -> # vars
                    obj.v(i,1).n = numel(E{i});
                    % dm -> dimensions of vars
                    obj.v(i,1).dm = nan(obj.v(i,1).n,1);
                    for p = 1:obj.v(i,1).n
                        obj.v(i,1).dm(p) = numel(E{i}{p});
                    end
                    % E[X_{i}]
                    obj.v(i,1).E = E{i};
                    % Var[X_{i}]
                    obj.v(i,1).Var = Cov{i,i};
                    % Cov[X_{i},Pa(X_{i})]
                    obj.v(i,1).Cov = Cov(i,obj.v(i,1).Pa);
                end
                % labels
                if nargin>3
                    obj.v(i,1).Lbl = Lbl{i};
                end
                % data
                if nargin>4
                    obj.v(i,1).Z = Z{i};
                end
            end
        end % DAG
        
        %% Adjacency matrix
        function A = AdjacencyMatrix(obj)
            % AdjacencyMatrix: compute undirected adjacency matrix
            % associated with Graph
            % 
            % INPUTS:
            % - obj: [1 x 1] DAG object.
            % OUTPUTS:
            % - A: [n x n] array: adjacency matrix, containing 1 for
            % adjacent vertex pairs, 0 for non-adjacent vertex pairs.
            
            % init
            A = zeros(obj.n,obj.n);
            % loop over nodes
            for i = 1:obj.n
                A(i,obj.v(i,1).Pa) = 1;
            end
            % fill out transpose
            A = A + A';
        end % AdjacencyMatrix
        
        %% Numerically compute covariances
        function Cov = CovarianceNumerical(obj,A,plen)
            % CovarianceNumerical: compute relevant covariances from the
            % original trajetory samples.
            % 
            % INPUTS:
            % - A: [n x n] array: adjacency matrix for the graph.
            % - plen: [1 x 1] scalar: maximum path length for computing
            %   covariances.
            % OUTPUTS:
            % - Cov: [n x n] cell array: covariances between paris of
            %   vertices (with appropriate sub-division into variables).
            
            % output
            fprintf('Computing covariances...\n')
            
            % initialise Cov
            Cov = cell(obj.n,obj.n);
            % loop over nodes v_{i}, i = 1,2,...,n
            for i = 1:obj.n
                % Cov[v_{i},v_{i}]
                Cov{i,i} = obj.v(i).Var;
                % Cov[v_{i},Pa(v_{i})]
                if ~isempty(obj.v(i).Pa)
                    for p = 1:numel(obj.v(i).Pa)
                        Cov{i,obj.v(i).Pa(p)} = obj.v(i).Cov{p};
                        Cov{obj.v(i).Pa(p),i} = mat2cell(cell2mat(obj.v(i).Cov{p})',...
                                    obj.v(obj.v(i).Pa(p)).dm,obj.v(i).dm);
                    end % p
                end
                % other covariances
                % initialise A
                pA = A;
                % loop over path lengths
                for lg = 1:plen
                    % find paths of this length
                    edg = find(pA(i,1:(i-1))>0);
                    for k = 1:numel(edg)
                        % j -> current node
                        j = edg(k);
                        % Cov[v_{i},v_{j}]
                        Cov{i,j} = mat2cell(obj.v(i).Z*(obj.v(j).Z')./size(obj.v(i).Z,2) -...
                                        cell2mat(obj.v(i).E)*cell2mat(obj.v(j).E)',...
                                            obj.v(i).dm,obj.v(j).dm);
                        Cov{j,i} = mat2cell(cell2mat(Cov{i,j})',...
                                            obj.v(j).dm,obj.v(i).dm);
                    end % k
                    pA = pA*A;
                end % lg
            end % i
        end % CovarianceNumerical
        
        %% Moralize the graph
        function A = Moralize(obj)
            % Moralize: compute adjacency matrix for the moral graph
            % associated with the DAG
            %
            % INPUTS:
            % - obj: [1 x 1] DAG object.
            % OUTPUTS:
            % - A: [n x n] array: adjacency matrix for the moralized graph.
            
            % output
            fprintf('Starting moralization...\n')
            
            % compute undirected adj. matrix
            A = obj.AdjacencyMatrix();
            
            % loop over nodes
            for i = 1:obj.n
                % get node parent list
                Pai = obj.v(i).Pa;
                % loop over parents
                for j = 1:numel(Pai)
                    for k = (j+1):numel(Pai)
                        % if there's not already a link between the
                        % parents, add one
                        if A(Pai(j),Pai(k))~=1
                            A(Pai(j),Pai(k)) = 1;
                            A(Pai(k),Pai(j)) = 1;
                        end
                    end % k
                end % j
                % output
                if rem(i,10)==0
                    fprintf('#')
                    if rem(i,100)==0
                        fprintf('\n')
                    end
                end
            end % i
            
            % output
            fprintf('Moralization computed.\n')
        end % Moralize
        
        %% Triangulate the moralized graph
        function A = Triangulate(obj)
            % Triangulate: compute the adjacency matrix for the
            % triangulated, moral graph associated with the DAG.
            % (Elimination game: Heggernes [2005])
            %
            % INPUTS:
            % - obj: [1 x 1] DAG object.
            % OUTPUTS:
            % - A: [n x n] array: adjacency matrix for the triangulated,
            %   moralized graph.
            
            % output
            fprintf('Starting triangulation...\n')
            
            % compute the moral graph
            A = obj.Moralize();
            % add in 'self-edges'
            A = A + eye(obj.n);
            
            if 1
                idx = [(2:obj.n)';1];
            else
                idx = (1:obj.n)';
            end
            
            % initialise
            % seq. of sub-graphs G^{i}
            Gi = A(idx,idx);
            % seq. of nodes introduce to turn neighbourhood into a clique
            cmFi = zeros(obj.n,obj.n);
            
            % loop over vertices
            for i = 1:obj.n
                % get neighbourhood N_{G^{i}}(v_i)
                Ni = logical(Gi(i,:));
                % find deficiency D_{G^{i}}(v_i)
                Di = 1-Gi(Ni,Ni);
                % F^{i} = D_{G^{i}}(v_i)
                Fi = zeros(obj.n,obj.n);
                Fi(Ni,Ni) = Di;
                cmFi = cmFi + Fi;
                % get G^{i+1} by adding vertices F^{i} to G^{i} and removing v_i
                Gip = Gi + Fi;
                Gip(i,:) = zeros(1,obj.n);
                Gip(:,i) = zeros(obj.n,1);
                % re-set 
                Gi = Gip;
                % output
                if rem(i,10)==0
                    fprintf('#')
                    if rem(i,100)==0
                        fprintf('\n')
                    end
                end
            end % i
            
            % add in union of F^{i} edges
            rA = A(idx,idx);
            rA = rA + cmFi;
            % restore
            A = nan(obj.n,obj.n);
            A(idx,idx) = rA;
            % remove repeated edges
            A = A>0;
            % remove diagonal
            A = A - eye(obj.n);
            
            % output
            fprintf('Triangulation computed.\n')
        end % Triangulate
        
        %% LB-triangulation algorithm (Heggerness et al., [2010])
        function A = LBTriangulate(obj)
            % LBTriangulate: Lekkerkerker and Boland triangulation of the
            % DAG.
            % 
            % INPUTS:
            % - obj: [1 x 1] DAG object.
            % OUTPUTS:
            % - A: [n x n] array: adjacency matrix for the triangulated,
            %   moralized graph.
            
            % output
            fprintf('Starting triangulation...\n')
            
            % compute the moral graph
            A = obj.Moralize();
            % add in 'self-edges'
            A = A + eye(obj.n);
            
            if 1
                idx = [(2:obj.n)';1];
            else
                idx = (1:obj.n)';
            end
            
            % initialise
            % seq. of sub-graphs G^{i}
            Gi = A(idx,idx);
            % seq. of nodes introduce to turn neighbourhood into a clique
            cmFi = zeros(obj.n,obj.n);
            
            % loop over vertices
            for i = 1:obj.n
                % get neighbourhood N_{G^{i}}(v_i)
                Ni = logical(Gi(i,:));
                % find G^{i}(V-N_{G^{i}}[v_{i}])
                GmNi = Gi;
                GmNi(Ni,:) = zeros(sum(Ni),obj.n);
                GmNi(:,Ni) = zeros(obj.n,sum(Ni));
                % find connected components
                gp_GmNi = graph(GmNi);
                C = conncomp(gp_GmNi);
                C(Ni) = 0;
                uqC = unique(C);
                % identify S(v_i) (substars), and get F (edges needed to
                % add to G^{i} in order to saturate substars)
                S = cell(numel(uqC),1);
                F = zeros(obj.n,obj.n);
                for iC = 2:numel(uqC)
                    % identify conn. comp. members
                    icmp = (C')==uqC(iC);
                    incmp = ~icmp;
                    icmp = find(icmp);
                    incmp = find(incmp);
                    % identify substars
                    S{iC} = incmp((sum(Gi(icmp,incmp),1)')>0);
                    % get
                    if ~isempty(S{iC})
                        F(S{iC},S{iC}) = 1-Gi(S{iC},S{iC});
                    end
                end
                % G^{i+1} = G^{i} + F
                Gi = Gi + F;
                % output
                if rem(i,10)==0
                    fprintf('#')
                    if rem(i,100)==0
                        fprintf('\n')
                    end
                end
            end % i
            
            % restore
            A = nan(obj.n,obj.n);
            A(idx,idx) = Gi;
            % remove repeated edges
            A = A>0;
            % remove diagonal
            A = A - eye(obj.n);
            
            % output
            fprintf('Triangulation computed.\n')
        end
        
        %% Maximum cardinality search
        function [lbl,ord] = MaxCardinality(obj,A)
            % MaxCardinality: carry out maximum cardianlity search, and
            % report node labelling
            %
            % INPUTS:
            % - A: [n x n] array: adjacency matrix associated with the
            %   triangulated moral graph.
            % OUTPUTS:
            % - lbl: [n x 1] integer array: label associated with each of
            %   the vertices in the DAG.
            % - ord: [n x 1] integer array: original variable indices in
            %   DAG ordered according to the labelling found.
            
            % init
            ord = nan(obj.n,1);
            lbl = nan(obj.n,1);
            % start at node 1
            ord(1) = 1;
            lbl(1) = 1;
            % loop over nodes
            for i = 1:(obj.n-1)
                % find unlabelled nodes
                unlb = setdiff((1:obj.n)',ord(1:i));
                % find # labelled neighbours of all unlabelled nodes
                n_lb = sum(A(unlb,ord(1:i)),2);
                max_n_lb = max(n_lb);
                % choose lowest numbered unlabelled node
                i_lb = unlb(find(n_lb==max_n_lb,1,'first'));
                ord(i+1) = i_lb;
                lbl(i_lb) = i+1;
                % check whether all labelled nodes of this node are
                % neighbours of each other
                if 1
                    % find neighbours
                    c_ng = find(A(i_lb,:));
                    % find labelled neighbours
                    c_lb_ng = c_ng(ismember(c_ng,ord(1:i)));
                    n_lb_ng = numel(c_lb_ng);
                    % get sub-graph
                    ng_A = A(c_lb_ng,c_lb_ng);
                    % check
                    if sum(ng_A(:))~=(n_lb_ng*(n_lb_ng-1))
                        warning('Not all neighbours are neighbours.')
                    end
                end
            end % i
        end % MaxCardinality
        
        %% Find the cliques
        function [C,A] = Cliques(obj)
            % Cliques: find the cliqies associated with the DAG
            % (Algorithm due to Bron & Kerbosch [1973])
            % 
            % INPUTS:
            % - obj: [1 x 1] DAG object.
            % OUTPUTS:
            % - C: [N_Clq x 1] cell array: list of cliques: each cell
            %   contains the indices of the variables associated with the
            %   clique.
            % - A: [n x n] array: adjacency matrix for triangulated moral
            %   graph.
            
            % output
            fprintf('Starting clique finder...\n')
            
            % get adj. matrix for triangulated moral graph
            A = obj.LBTriangulate();
            
            % initialise
            R = []; X = [];
            P = (1:1:obj.n)';
            
            % get cliques
            % init
            C = [];
            % run
            C = obj.BronKerbosch(R,P,X,C,A);
            
            % output
            fprintf('Cliques identified.\n')
        end % Cliques
        
        %% Recursive Bron-Kerbosch function
        function C = BronKerbosch(obj,R,P,X,C,A)
            % BronKerbosch: recursive sub-function for the Bron Kerbosch
            % clique identification algorithm.
            % 
            % INPUTS:
            % - 
            % OUTPUTS:
            % - 
            
            % # vertices
            n  = numel(P);
            % initialise clique storage
            
            if isempty([P;X])
                % add R to C
                C = [C;{R}];
            else
                % loop over vertices
                for i = 1:n
                    % find N(v_{i})
                    Ni = find(A(P(1),:))';
                    % find R \union v_{i}
                    Ri = unique([R;P(1)]);
                    % find P \intersection N(v_{i})
                    Pi = intersect(P,Ni);
                    % find X \intersection N(v_{i})
                    Xi = intersect(X,Ni);
                    
                    % step down
                    C = obj.BronKerbosch(Ri,Pi,Xi,C,A);
                    
                    % set X <- X \union v_i
                    X = unique([X;P(1)]);
                    % set P <- P \ v_i
                    P(1) = [];
                end % i
            end
        end % BronKerbosch
    end % methods (Static)
end % DAG