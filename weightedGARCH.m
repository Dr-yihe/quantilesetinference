function [mdl_opt,theta_opt,f_opt] = weightedGARCH(y)

y=y(~isnan(y));
T=length(y);
C=quantile(y,0.9);
x=abs(y).*(y>C);
w=nan(T,1);w(1)=1;
for i=2:T
    w(i)=sum(x((i-1):-1:1)./((1:(i-1)).^9)')/C;
end
w=(max(w,1)).^(-4);

garch11=garch('Offset',NaN,'GARCHLags',1,'ARCHLags',1);
mdl=estimate(garch11,y,'Display','Off');
tol=0.999;
%theta_mdl=[mdl.Constant;cell2mat(mdl.ARCH);norminv((cell2mat(mdl.GARCH)/tol+1)/2)];
theta_mdl=[mdl.Constant;cell2mat(mdl.ARCH);cell2mat(mdl.GARCH);mdl.Offset];

options = optimoptions('fmincon','Display','off');
[theta_opt,f_opt]=fmincon(@(theta)(swGARCH11fun(y,theta,w)),theta_mdl,[],[],[],[],[-inf,-inf,-tol,-inf],[inf,inf,tol,inf],[],options);

mdl_opt=garch('Constant',theta_opt(1),'ARCH',theta_opt(2),'GARCH',theta_opt(3),'Offset',theta_opt(4));
end

