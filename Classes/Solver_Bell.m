classdef Solver_Bell
    % Solver for the bell-tower system of ODEs
    
    properties (SetAccess = private)
        Prm     % Fixed solver params
        Nms     % Input names
    end
    
    methods
        %% Constructor
        function obj = Solver(ui)
            obj.Prm = ui.Prm;
            obj.Nms = ui.Nms;
        end
        %% Derivative function
        % du_i/dt(t) = f_i(t,u(t),\theta)
        function [du,A,b] = f(obj,phi)
            % # data inputs
            nD = size(phi.th,2);
            % storage
            du.ddx = nan(2,nD);
            du.ddth = nan(obj.Prm.B.n,nD);
            for iD = 1:nD
                %----------------------
                % d^{2}\phi_i/dt^{2}
                % A
                cA = cell(2,2);
                % d^{2}x/dt^{2} equations
                cA{1,1} = eye(2);
                cA{1,2} = bsxfun(@times,obj.Prm.B.drc',...
                    ((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*cos(phi.th(:,iD)))');
                % d^{2}\theta{}/dt^{2}
                cA{2,1} = bsxfun(@times,cos(phi.th(:,iD)),obj.Prm.B.drc);
                cA{2,2} = diag(obj.Prm.B.l);
                % b
                cb = cell(2,1);
                % d^{2}x/dt^{2} equations
                cb{1} = obj.Prm.B.drc'*((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*...
                    (phi.dth(:,iD).^2).*cos(phi.th(:,iD))) -...
                    2.*phi.lam(:,iD).*phi.dx(:,iD) - (phi.wg(:,iD).^2).*phi.x(:,iD);
                % d^{2}\theta{}/dt^{2} equations
                cb{2} = -obj.Prm.g.*sin(phi.th(:,iD));
                %----------------------
                % solve
                A = cell2mat(cA); b = cell2mat(cb);
                v = A\b;
                % d^{2}x/dt^{2}
                du.ddx(:,iD) = v(1:2);
                % d^{2}\theta{}/dt^{2}
                du.ddth(:,iD) = v(3:end);
            end
            % degree conversion
            du.ddth = du.ddth;
        end
        %% Higher-order derivative function
        function du = df(obj,phi)
            % # data inputs
            nD = size(phi.th,2);
            % storage
            du.ddx = nan(2,nD);
            du.ddth = nan(obj.Prm.B.n,nD);
            du.dddx = nan(2,nD);
            du.dddth = nan(obj.Prm.B.n,nD);
            du.ddddx = nan(2,nD);
            du.ddddth = nan(obj.Prm.B.n,nD);
            % loop over data
            for iD = 1:nD
                %----------------------
                % f(u,\theta)
                % A
                cA = cell(2,2);
                % x eq.
                cA{1,1} = eye(2);
                cA{1,2} = bsxfun(@times,obj.Prm.B.drc',...
                    ((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*cos(phi.th(:,iD)))');
                % \theta eq.
                cA{2,1} = bsxfun(@times,cos(phi.th(:,iD)),obj.Prm.B.drc);
                cA{2,2} = diag(obj.Prm.B.l);
                % b
                cb = cell(2,1);
                % x eq.
                cb{1} = obj.Prm.B.drc'*((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*...
                    (phi.dth(:,iD).^2).*cos(phi.th(:,iD))) -...
                    2.*phi.lam(:,iD).*phi.dx(:,iD) - (phi.wg(:,iD).^2).*phi.x(:,iD);
                % \theta eq.
                cb{2} = -obj.Prm.g.*sin(phi.th(:,iD));
                % f
                A = sparse(cell2mat(cA)); b = cell2mat(cb);
                iA = A\eye(size(A));
                fv = iA*b;
                du.ddx(:,iD) = fv(1:2);
                du.ddth(:,iD) = fv(3:end);
                %----------------------
                % df/dt(u,\theta)
                % dA/dt
                cdA = cell(2,2);
                % x eq.
                cdA{1,1} = zeros(2,2);
                cdA{1,2} = -bsxfun(@times,obj.Prm.B.drc',...
                    ((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*...
                    phi.dth(:,iD).*sin(phi.th(:,iD)))');
                % \theta eq.
                cdA{2,1} = -bsxfun(@times,obj.Prm.B.drc,phi.dth(:,iD).*sin(phi.th(:,iD)));
                cdA{2,2} = zeros(obj.Prm.B.n);
                % db/dt
                cdb = cell(2,1);
                % x eq.
                cdb{1} = obj.Prm.B.drc'*...
                    ((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*...
                    (2.*phi.dth(:,iD).*du.ddth(:,iD).*sin(phi.th(:,iD)) +...
                    (phi.dth(:,iD).^3).*cos(phi.th(:,iD)))) -...
                    2.*phi.lam(:,iD).*du.ddx(:,iD) - (phi.wg(:,iD).^2).*phi.dx(:,iD);
                % \theta eq.
                cdb{2} = -obj.Prm.g.*phi.dth(:,iD).*cos(phi.th(:,iD));
                % df/dt
                dA = cell2mat(cdA); db = cell2mat(cdb);
                dfv = iA*(db - dA*fv);
                du.dddx(:,iD) = dfv(1:2);
                du.dddth(:,iD) = dfv(3:end);
                %----------------------
                % d^{2}f/dt^{2}(u,\theta)
                % d^{2}A/dt^{2}
                cddA = cell(2,2);
                % x eq.
                cddA{1,1} = zeros(2,2);
                cddA{1,2} = -bsxfun(@times,obj.Prm.B.drc',(obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M)'.*...
                               (du.ddth(:,iD).*sin(phi.th(:,iD)) + (phi.dth(:,iD).^2).*cos(phi.th(:,iD)))');
                % \theta eq.
                cddA{2,1} = -bsxfun(@times,obj.Prm.B.drc,...
                                du.ddth(:,iD).*sin(phi.th(:,iD)) + (phi.dth(:,iD).^2).*cos(phi.th(:,iD)));
                cddA{2,2} = zeros(obj.Prm.B.n);
                % d^{2}b/dt^{2}
                cddb = cell(2,1);
                % x eq.
                cddb{1} = obj.Prm.B.drc'*((obj.Prm.B.m.*obj.Prm.B.rc./obj.Prm.C.M).*...
                                2.*(du.dddth(:,iD).*phi.dth(:,iD) + du.ddth(:,iD).^2).*sin(phi.th(:,iD)) +...
                                5.*du.ddth(:,iD).*(phi.dth(:,iD).^2).*cos(phi.th(:,iD)) - (phi.dth(:,iD).^4).*sin(phi.th(:,iD))) -...
                            2.*phi.lam(:,iD).*du.dddx(:,iD) - (phi.wg(:,iD).^2).*du.ddx(:,iD);
                % \theta eq.
                cddb{2} = -obj.Prm.g.*(du.ddth(:,iD).*cos(phi.th(:,iD)) - (phi.dth(:,iD).^2).*sin(phi.th(:,iD)));
                % d^{2}f/dt^{2}
                ddA = cell2mat(cddA); ddb = cell2mat(cddb);
                ddfv = iA*(ddb - ddA*fv - 2.*dA*dfv);
                du.ddddx(:,iD) = ddfv(1:2);
                du.ddddth(:,iD) = ddfv(3:end);
            end
        end
        %% Euler evolution function
        % \hat{u}_i(t) = F_{i}(t,u(t),\theta)
        function uh = Fel(obj,u,t0,t1,xi)
            % cat structures
            phi = u;
            xi_fld = fields(xi);
            for iF = 1:numel(xi_fld)
                phi.(xi_fld{iF}) = xi.(xi_fld{iF});
            end
            phi.dt = t1-t0;
            %----------------------
            % Evaluate f(.)
            f = obj.f(phi);
            uh = phi;
            %----------------------
            % \hat{dx/dt}_i(t)
            uh.dx = phi.dx + bsxfun(@times,f.ddx,phi.dt);
            %----------------------
            % \hat{d\theta/dt}_j(t)
            uh.dth = phi.dth + bsxfun(@times,f.ddth,phi.dt);
            %----------------------
            % \hat{x}_i(t)
            uh.x = phi.x + bsxfun(@times,uh.dx,phi.dt);
            %----------------------
            % \hat{\theta}_j(t)
            uh.th = phi.th + bsxfun(@times,uh.dth,phi.dt);
            %----------------------
            % impose hard stop at top of swing
            % >180
            uh.dth(uh.th>pi) = -uh.dth(uh.th>pi);
            uh.th(uh.th>pi) = pi;
            % <-180
            uh.dth(uh.th<-pi) = -uh.dth(uh.th<-pi);
            uh.th(uh.th<-pi) = -pi;
        end
        %% Second-order Runge-Kutta evolution function
        function uh = Frk2(obj,phi)
            %----------------------
            % Evaluate components
            % constants
            a21 = 1/2;
            c2 = 1/2; b2 = 1;
            h = phi.dt;
            % \xi_1
            xi1 = phi;
            % \xi_2
            fx1 = obj.f(xi1);
            xi2.dx = phi.dx + a21.*bsxfun(@times,fx1.ddx,h);
            xi2.x = phi.x + a21.*bsxfun(@times,xi1.dx,h);
            xi2.dth = phi.dth + a21.*bsxfun(@times,fx1.ddth,h);
            xi2.th = phi.th + a21.*bsxfun(@times,xi1.dth,h);
            xi2.lam = phi.lam;
            xi2.wg = phi.wg;
            %----------------------
            % Approximation
            % initialise
            uh = phi;
            % evolve \xi_2
            fx2 = obj.f(xi2);
            % update
            uh.dx = uh.dx + b2.*bsxfun(@times,fx2.ddx,h);
            uh.x = uh.x + b2.*bsxfun(@times,xi2.dx,h);
            uh.dth = uh.dth + b2.*bsxfun(@times,fx2.ddth,h);
            uh.th = uh.th + b2.*bsxfun(@times,xi2.dth,h);
        end
        %% Second-order evolution function
        function uh = Fel2(obj,phi)
            %----------------------
            % Evaluate f(.) and derivatives
            df = obj.df(phi);
            uh = phi;
            %----------------------
            % \hat{dx/dt}_i(t)
            uh.dx = phi.dx + bsxfun(@times,df.ddx,phi.dt) +...
                             bsxfun(@times,df.dddx,phi.dt.^2).*(1/2);
            %----------------------
            % \hat{d\theta/dt}_j(t)
            uh.dth = phi.dth + bsxfun(@times,df.ddth,phi.dt) +...
                               bsxfun(@times,df.dddth,phi.dt.^2).*(1/2);
            %----------------------
            % \hat{x}_i(t)
            uh.x = phi.x + bsxfun(@times,uh.dx,phi.dt) +...
                           bsxfun(@times,df.ddx,phi.dt.^2).*(1/2);
            %----------------------
            % \hat{\theta}_j(t)
            uh.th = phi.th + bsxfun(@times,uh.dth,phi.dt) +...
                             bsxfun(@times,df.ddth,phi.dt.^2).*(1/2);
            %----------------------
            % impose hard stop at top of swing
            if any(uh.th(:)>pi)||any(uh.th(:)<-pi)
                pi
            end
            % >180
            uh.dth(uh.th>pi) = -uh.dth(uh.th>pi);
            uh.th(uh.th>pi) = pi;
            % <-180
            uh.dth(uh.th<-pi) = -uh.dth(uh.th<-pi);
            uh.th(uh.th<-pi) = -pi;
        end
    end
    
end

