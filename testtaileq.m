function [h,pval] = testtaileq(x,k,tau)
if nargin<3
    tau=0.95;
end
[gam,~,~,thres]=nanGammaEst(sort(x),k,'Hill');
gamsort=sort(gam);
Tstat=sqrt(k)*(gamsort(end)-gamsort(1))/mean(gam);
z=(x>ones(size(x,1),1)*thres)+0;
Sigma=z'*z/k;
Z=mvnrnd(zeros(size(Sigma,1),1),Sigma,5000);
W=max(Z,[],2)-min(Z,[],2);
h=Tstat>quantile(W,tau);
pval=mean(W>Tstat);
end

