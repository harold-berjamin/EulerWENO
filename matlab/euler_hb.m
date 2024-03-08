% clear;
format long

% Job parameters
plots = 1; % 0, 1, 2, 3
flux = @Osher; % @LxF, @Osher
inter = @WENO_Roe; % @None, @WENO, @WENO_Roe
integ = @RK4; % @RK1, @RK3, @RK4
BC = @OutgoingBC; % @OutgoingBC, @PeriodicBC
u0 = @Riemann; % @Density, @Riemann
% <!> @RK4 requires smaller time-steps by a factor 2/3 (cf. CFL section below)

% Mesh size, final time
xlims = [-5, 5];
Nx = 100;
Tf = 2;

global gam % heat capacity ratio
gam = 1.4;

% Data initialisation
global dt dx
t = 0;
n = 0;
dx = abs( (xlims(end)-xlims(1))/Nx );
x = linspace(xlims(1)-2.5*dx,xlims(end)+2.5*dx,Nx+6);
Nx = Nx+6;

% Initial condition averaging
u = BC(cellav(u0,x,dx));

% CFL
Co = 0.6; % 0.6
[vals, ~] = EigA(u);
amax = max(max(abs(vals)));
dt = Co * dx/amax;

% Graphics initialisation
if plots>0
    figure(1);
    clf;
    hp = plot(x,u(plots,:),'b.-');
    xlim([x(1) x(end)]);
    M = 1.1*(max(abs(u(plots,:))) + 0.02);
    ylim([-M, M]);
    ht = title(strcat('t = ',num2str(t)));
end

% Main loop
tStart = tic;
tic
while t<Tf
    % Iteration
    u = integ(u,flux,inter,BC);
    t = t + dt;
    n = n + 1;
    
    % CFL
    [vals, ~] = EigA(u);
    amax = max(max(abs(vals)));
    dt = Co * dx/amax;
    
    % Graphics update
    if (plots>0) && (toc>1)
        % intermediate solution
        set(hp,'YData',u(plots,:));
        set(ht,'String',strcat('t = ',num2str(t)));
        drawnow;
        tic
    end
end
tEnd = toc(tStart);
disp(strcat('Terminated in~',num2str(tEnd),' seconds.'));
disp(strcat('Terminated in~',num2str(n),' iterations.'));

% Exact solution
if isequal(u0,@Density)
    utheo = @(x) u0(x-t);
elseif isequal(u0,@Riemann)
    UJ = [u0(x(1)), u0(x(end))];
    utheo = RiemannExact(UJ,gam,t);
end

% Plot final solution
if plots>0
    % numerical solution
    set(hp,'YData',u(plots,:));
    set(ht,'String',strcat('t = ',num2str(t)));
    drawnow;
    hold on
    % exact solution
    xplot = linspace(x(1),x(end),max([2*Nx,1e3]));
    uth = utheo(xplot);
    plot(xplot,uth(plots,:),'k-');
end

% Error wrt. exact solution
if isequal(u0,@Density)
    uth = cellav(utheo, x, dx);
    ierr = find((x>xlims(1)).*(x<xlims(end)));
    derr = u(:,ierr) - uth(:,ierr);
    one_err = [norm(derr(1,:)*dx,1), norm(derr(2,:)*dx,1), norm(derr(3,:)*dx,1)];
    two_err = [norm(derr(1,:)*dx,2), norm(derr(2,:)*dx,2), norm(derr(3,:)*dx,2)];
    inf_err = [norm(derr(1,:),Inf), norm(derr(2,:),Inf), norm(derr(3,:),Inf)];
    disp([one_err; two_err; inf_err]);
end

%% functions

% Euler equations physics
% physical flux u_t + f(u)_x = 0     (Chap. 3 p. 87, Toro, 2009)

function flx = f(u)
    global gam
    v = u(2,:)./u(1,:);
    flx = [u(2,:); 0.5*(3-gam)*u(1,:).*v.^2 + (gam-1)*u(3,:); gam*u(3,:).*v - 0.5*(gam-1)*u(1,:).*v.^3];
end

function jac = A(u)
    global gam
    v = u(2,:)./u(1,:);
    p = (gam-1)*(u(3,:) - 0.5*u(1,:).*v.^2);
    a = sqrt(gam*p./u(1,:));
    N = length(u(1,:));
    jac = [zeros(1,N); 0.5*(gam-3)*v.^2; 0.5*(gam-2)*v.^3 - a.^2.*v/(gam-1); ...
        ones(1,N); (3-gam)*v; 0.5*(3-2*gam)*v.^2 + a.^2/(gam-1); ...
        zeros(1,N); (gam-1)*ones(1,N); gam*v];
end

function [vals, vecs] = EigA(u)
    global gam
    v = u(2,:)./u(1,:);
    p = (gam-1)*(u(3,:) - 0.5*u(1,:).*v.^2);
    a = sqrt(gam*p./u(1,:));
    vals = [v-a; v; v+a];
    H = 0.5*v.^2 + a.^2/(gam-1);
    N = length(u(1,:));
    vecs = [ones(1,N); v-a; H-v.*a; ...
        ones(1,N); v; 0.5*v.^2; ...
        ones(1,N); v+a; H+v.*a];
end

function mat = aA(u)
    [vals, vecs] = EigA(u);
    N = length(u(1,:));
    mat = zeros(9,N);
    for i=1:N
        aLamb = diag(abs(vals(:,i)));
        R = reshape(vecs(:,i),3,3);
        aA = R*aLamb*inv(R);
        mat(:,i) = reshape(aA,9,1);
    end
end

% Initial conditions

function u0 = Density(x) % Density perturbation
    global gam
    rho = 1 + 0.2*sin(pi*x);
    u = ones(1,length(x));
    p = ones(1,length(x));
    u0 = [rho; rho.*u; 0.5*rho.*u.^2 + p./(gam-1)];
end

function u0 = Riemann(x) % Riemann jump L - R
    global gam
    % Lax
    rhoJ = [0.445, 0.5];
    uJ = [0.698, 0];
    pJ = [3.528, 0.571];
    % Sod
    rhoJ = [1, 0.125];
    uJ = [0, 0];
    pJ = [1, 0.1];
    % Riemann problem selection below
    rhoJ = [1, 0.125];
    uJ = [0, 0];
    pJ = [1, 0.1];
    % Initial data
    UJ = [rhoJ; rhoJ.*uJ; 0.5*rhoJ.*uJ.^2 + pJ./(gam-1)];
    u0 = UJ(:,1).*(x<0) + UJ(:,2).*(x>=0);
end

function u = cellav(u0,x,dx) % cell averages
    u = u0(x);
    uav0 = @(x) integral(u0,x-0.5*dx,x+0.5*dx,'ArrayValued',true) / dx;
    for i=1:length(x)
        u(:,i) = uav0(x(i));
    end
end

% Boundary conditions

function v = PeriodicBC(u) % (periodic)
    Nx = length(u(1,:));
    v = u;
    v(:,1) = u(:,Nx-5);
    v(:,2) = u(:,Nx-4);
    v(:,3) = u(:,Nx-3);
    v(:,Nx-2) = u(:,4);
    v(:,Nx-1) = u(:,5);
    v(:,Nx)   = u(:,6);
end

function v = OutgoingBC(u) % (outgoing)
    Nx = length(u(1,:));
    v = u;
    v(:,1) = u(:,4);
    v(:,2) = u(:,4);
    v(:,3) = u(:,4);
    v(:,Nx-2) = u(:,Nx-3);
    v(:,Nx-1) = u(:,Nx-3);
    v(:,Nx)   = u(:,Nx-3);
end

% First-order schemes

function [up05m, up05p] = None(u)
    up05m = u;
    up05p = circshift(u, -1, 2);
end

function fp05 = LxF(u,inter) % Lax-Friedrichs flux
    [up05m, up05p] = inter(u);
    [vals, ~] = EigA(u);
    amax = max(abs(vals(3,:)-vals(2,:)) + abs(vals(2,:)));
    fp05 = 0.5*(f(up05m) + f(up05p) - amax*(up05p - up05m));
end

function fp05 = Osher(u,inter) % Osher flux
    [up05m, up05p] = inter(u);
    du = up05p - up05m;
    u1 = up05m + (0.5-sqrt(15)/10)*du;
    u2 = up05m + 0.5*du;
    u3 = up05m + (0.5+sqrt(15)/10)*du;
    iA = (5*aA(u1) + 8*aA(u2) + 5*aA(u3))/18;
    N = length(du(1,:));
    fp05 = zeros(3,N);
    for i=1:N
        iAr = reshape(iA(:,i),3,3);
        fp05(:,i) = 0.5*(f(up05m(:,i)) + f(up05p(:,i)) - iAr*du(:,i));
    end
end

function up = RK1(u,flux,inter,BC)
    global dt dx
    fp05 = flux(u,inter);
    fm05 = circshift(fp05, +1, 2);
    up = BC(u - dt/dx*(fp05 - fm05));
end

% High-order WENO schemes

function up05m = reconstructWENO(uim2,uim1,ui,uip1,uip2)  
    % polynomial approx i+1/2^-
    up051 = (2*uim2 - 7*uim1 + 11*ui )/6; % p0
    up052 = ( -uim1 + 5*ui   + 2*uip1)/6; % p1
    up053 = (2*ui   + 5*uip1 -   uip2)/6; % p2
    % smoothness indicators
    b1 = 13/12*(uim2 - 2*uim1 + ui  ).^2 + 0.25*(uim2 - 4*uim1 + 3*ui).^2;
    b2 = 13/12*(uim1 - 2*ui   + uip1).^2 + 0.25*(uim1 - uip1).^2;
    b3 = 13/12*(ui   - 2*uip1 + uip2).^2 + 0.25*(3*ui - 4*uip1 + uip2).^2;
    % weights
    w1 = 0.1 ./ (1e-6 + b1).^2;
    w2 = 0.6 ./ (1e-6 + b2).^2;
    w3 = 0.3 ./ (1e-6 + b3).^2;
    ws = w1 + w2 + w3;
    % reconstructed cell-interface value
    up05m = (w1.*up051 + w2.*up052 + w3.*up053)./ws;
end

function [up05m, up05p] = WENO(u)
    Nx = length(u(1,:));
    up05m = u;
    up05p = u;
    % reconstructed cell-interface values
    for i=1:3
        for j=3:Nx-3
            up05m(i,j) = reconstructWENO(u(i,j-2),u(i,j-1),u(i,j),u(i,j+1),u(i,j+2));
            up05p(i,j) = reconstructWENO(u(i,j+3),u(i,j+2),u(i,j+1),u(i,j),u(i,j-1));
        end
    end
end

function [P, Pm] = Roe(u, up1) % Roe average
    global gam
    sRhoL = sqrt(u(1));
    sRhoR = sqrt(up1(1));
    vL = u(2)./u(1);
    vR = up1(2)./up1(1);
    v = (sRhoL*vL+sRhoR*vR)/(sRhoL+sRhoR);
    pL = (gam-1)*(u(3) - 0.5*u(1)*vL^2);
    pR = (gam-1)*(up1(3) - 0.5*up1(1)*vR^2);
    aL = sqrt(gam*pL/u(1));
    aR = sqrt(gam*pR/up1(1));
    HL = 0.5*vL^2 + aL^2/(gam-1);
    HR = 0.5*vR^2 + aR^2/(gam-1);
    H = (sRhoL*HL+sRhoR*HR)/(sRhoL+sRhoR);
    h = H - 0.5*v^2;
    a = sqrt( (gam-1)*h );
    vecsR = [1; v-a; H-v*a; ...
        1; v; 0.5*v^2; ...
        1; v+a; H+v*a];
    vecsL = 0.5/h* [0.5*v^2+v*h/a; 2*h-v^2; 0.5*v^2-v*h/a; ...
        -h/a-v; 2*v; h/a-v; ...
        1; -2; 1];
    P = reshape(vecsR,3,3);
    Pm = reshape(vecsL,3,3);
end

function [up05m, up05p] = WENO_Roe(u)
    Nx = length(u(1,:));
    up05m = u;
    up05p = circshift(u, -1, 2);
    w = u;
    for j=3:Nx-3
        % Roe average
        [P, Pm] = Roe(u(:,j), u(:,j+1));
        % local characteristic variables
        for k=j-2:j+3
            w(:,k) = Pm * u(:,k);
        end
        % reconstructed cell-interface values
        wp05m = w(:,j);
        wp05p = w(:,j+1);
        for i=1:3
            wp05m(i) = reconstructWENO(w(i,j-2),w(i,j-1),w(i,j),w(i,j+1),w(i,j+2));
            wp05p(i) = reconstructWENO(w(i,j+3),w(i,j+2),w(i,j+1),w(i,j),w(i,j-1));
        end
        % physics variables
        up05m(:,j) = P * wp05m;
        up05p(:,j) = P * wp05p;
    end
end

function up = RK3(u,flux,inter,BC)
    global dt dx
    % 1
    fp05 = flux(u,inter);
    fm05 = circshift(fp05, +1, 2);
    u1 = BC(u - dt/dx*(fp05 - fm05));
    %2
    fp05 = flux(u1,inter);
    fm05 = circshift(fp05, +1, 2);
    u2 = BC(0.25*(3*u + u1 - dt/dx*(fp05 - fm05)));
    %3
    fp05 = flux(u2,inter);
    fm05 = circshift(fp05, +1, 2);
    up = BC((u + 2*u2 - 2*dt/dx*(fp05 - fm05))/3);
end

function up = RK4(u,flux,inter,BC)
    global dt dx
    % 1
    fp05 = flux(u,inter);
    fm05 = circshift(fp05, +1, 2);
    u1 = BC(u - 0.5*dt/dx*(fp05 - fm05));
    %2
    fp05 = flux(u1,inter);
    fm05 = circshift(fp05, +1, 2);
    u2 = BC(u - 0.5*dt/dx*(fp05 - fm05));
    %3
    fp05 = flux(u2,inter);
    fm05 = circshift(fp05, +1, 2);
    u3 = BC(u - dt/dx*(fp05 - fm05));
    %4
    fp05 = flux(u3,inter);
    fm05 = circshift(fp05, +1, 2);
    up = BC((-u + u1 + 2*u2 + u3 - 0.5*dt/dx*(fp05 - fm05))/3);
end