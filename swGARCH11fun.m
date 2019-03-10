function [L,eta] = swGARCH11fun(y,theta,w)
% Input parameter
% X: time series observation
% theta: parameter

T=length(y);
alpha_m=theta(1); % ensure it is positive
alpha=theta(2);
%tol=0.999;
%beta=(2*normcdf(theta(3))-1)*tol; % make sure no unit root
beta=theta(3);
if length(theta)>3
    mu=theta(4);
else
    mu=0;
end

h=nan(T+1,1);
e=nan(T+1,1);

e(2:end)=y-mu; e(1)=0;
h(1)=alpha_m/(1-beta);
for i=1:T
    h(i+1)=alpha_m+beta*h(i)+alpha*e(i)^2;
end
eta=e(2:end)./sqrt(h(2:end));
l=log(h(2:end))/2+abs(eta);
if nargin<3
    w=ones(T,1);
end
L=w'*l;


end

