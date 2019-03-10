function [gamma,bound,stdgamma,thres] = nanGammaEst(X,k,type)
gamma=zeros(1,size(X,2))*nan;
bound=gamma;stdgamma=gamma;thres=gamma;

for i=1:size(X,2)
    if k<1
        w=max(round(sum(~isnan(X(:,i)))*0.05),10);
    else
        w=k;
    end
    [gamma(i),bound(i),stdgamma(i),thres(i)]=GammaEst(X(~isnan(X(:,i)),i),w,type);
end


end

