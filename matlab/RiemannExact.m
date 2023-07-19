function sol = RiemannExact(UJ,gam,t)
% -------------------------------------------------------------------------
% Initial function arguments

rhoL = UJ(1,1);
rhouL = UJ(2,1);
EL = UJ(3,1);
rhoR = UJ(1,2);
rhouR = UJ(2,2);
ER = UJ(3,2);

uL = rhouL/rhoL;
pL = (gam-1)*(EL - 0.5*rhouL*uL);
uR = rhouR/rhoR;
pR = (gam-1)*(ER - 0.5*rhouR*uR);

% -------------------------------------------------------------------------
% useful quantities (cf. Toro p. 119)

aL = sqrt(gam*pL/rhoL);
aR = sqrt(gam*pR/rhoR);
AL = 2/((gam+1)*rhoL);
AR = 2/((gam+1)*rhoR);
BL = (gam-1)/(gam+1)*pL;
BR = (gam-1)/(gam+1)*pR;
du = uR - uL;

fL = @(p) (p-pL)*sqrt(AL/(p+BL)) .*(p>pL) ...
    + 2*aL/(gam-1)*((p/pL)^((gam-1)/(2*gam)) - 1) .*(p<=pL);
fR = @(p) (p-pR)*sqrt(AR/(p+BR)) .*(p>pR) ...
    + 2*aR/(gam-1)*((p/pR)^((gam-1)/(2*gam)) - 1) .*(p<=pR);
f = @(p) fL(p) + fR(p) + du;

% -------------------------------------------------------------------------
% Solution output

if uR-uL > 2*(aL+aR)/(gam-1)
    disp('Warning: Vacuum is created, i.e., pressure positivity is violated!')
else
    % star region
    p0 = max([eps, 0.5*(pL+pR)- du*(rhoL+rhoR)*(aL+aR)/8]);
    ps = fzero(f, p0);
    us = 0.5*(uR+uL) + 0.5*(fR(ps)-fL(ps));
    
    % solution types
    if ps>pL
        % disp('Left-going shock');
        rhoLs = rhoL * ( (gam-1)/(gam+1) + ps/pL )/( (gam-1)/(gam+1)*ps/pL + 1 );
        S1 = uL - aL*sqrt( (gam+1)/(2*gam)*ps/pL + (gam-1)/(2*gam) );
        rholeft = @(x) rhoL*(x<S1*t) + rhoLs*(x>=S1*t).*(x<us*t);
        uleft = @(x) uL*(x<S1*t) + us*(x>=S1*t).*(x<us*t);
        pleft = @(x) pL*(x<S1*t) + ps*(x>=S1*t).*(x<us*t);
    else
        % disp('Left-going rarefaction');
        aLs = aL + (uL-us)*(gam-1)/2;
        rhoLs = gam*ps/aLs^2;
        rholeft = @(x) rhoL*(x<(uL-aL)*t) + rhoLs*(x>=(us-aLs)*t).*(x<us*t) + rhoL*(2/(gam+1) + (gam-1)/((gam+1)*aL)*(uL-x/t)).^(2/(gam-1)).*(x>=(uL-aL)*t).*(x<(us-aLs)*t);
        uleft = @(x) uL*(x<(uL-aL)*t) + us*(x>=(us-aLs)*t).*(x<us*t) + 2/(gam+1)*(aL + (gam-1)/2*uL + x/t).*(x>=(uL-aL)*t).*(x<(us-aLs)*t);
        pleft = @(x) pL*(x<(uL-aL)*t) + ps*(x>=(us-aLs)*t).*(x<us*t) + pL*(2/(gam+1) + (gam-1)/((gam+1)*aL)*(uL-x/t)).^(2*gam/(gam-1)).*(x>=(uL-aL)*t).*(x<(us-aLs)*t);
    end
    if ps>pR
        % disp('Right-going shock');
        rhoRs = rhoR * ( (gam-1)/(gam+1) + ps/pR )/( (gam-1)/(gam+1)*ps/pR + 1 );
        S3 = uR + aR*sqrt( (gam+1)/(2*gam)*ps/pR + (gam-1)/(2*gam) );
        solR = @(x) [rhoR;uR;pR] * (x>S3*t) + [rhoRs;us;ps] * (x<=S3*t).*(x>=us*t);
        rhoright = @(x) rhoR*(x>S3*t) + rhoRs*(x<=S3*t).*(x>=us*t);
        uright = @(x) uR*(x>S3*t) + us*(x<=S3*t).*(x>=us*t);
        pright = @(x) pR*(x>S3*t) + ps*(x<=S3*t).*(x>=us*t);
    else
        % disp('Right-going rarefaction');
        aRs = aR + (us-uR)*(gam-1)/2;
        rhoRs = gam*ps/aRs^2;
        rhoright = @(x) rhoR*(x>(uR+aR)*t) + rhoRs*(x<=(us+aRs)*t).*(x>=us*t) + rhoR*(2/(gam+1) - (gam-1)/((gam+1)*aR)*(uR-x/t)).^(2/(gam-1)).*(x<=(uR+aR)*t).*(x>(us+aRs)*t);
        uright = @(x) uR*(x>(uR+aR)*t) + us*(x<=(us+aRs)*t).*(x>=us*t) + 2/(gam+1)*(-aR + (gam-1)/2*uR + x/t).*(x<=(uR+aR)*t).*(x>(us+aRs)*t);
        pright = @(x) pR*(x>(uR+aR)*t) + ps*(x<=(us+aRs)*t).*(x>=us*t) + pR*(2/(gam+1) - (gam-1)/((gam+1)*aR)*(uR-x/t)).^(2*gam/(gam-1)).*(x<=(uR+aR)*t).*(x>(us+aRs)*t);
    end
end

sol = @(x) [rholeft(x) + rhoright(x); ...
    rholeft(x).*uleft(x) + rhoright(x).*uright(x); ...
    0.5*rholeft(x).*uleft(x).^2 + pleft(x)/(gam-1) + 0.5*rhoright(x).*uright(x).^2 + pright(x)/(gam-1)];

end
