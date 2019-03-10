function [hest,hest_CR,hest_naive,dq,dq_naive,dq_CR,gamma_est,cv_1s] =Q_hat(X,p,k,u,tau)

% % Extreme Estimator of Quantile Region
% X: N*d input data
% k: intermediate sequence/ multiple input means auto selection
% p: probability (halfspace depth) level
% u: M*d matrix containing unit vectors (directions) representing a discretization of the unit
% sphere

if nargin<3
    error('not enough input');
elseif nargin<4
    u=randn(250*size(X,2),size(X,2)); % standard normal margins
    u=u./(sqrt(sum(u.^2,2))*ones(1,size(X,2))); % standardized to have unit norm
    u=[u;-u];
    tau=[];
elseif nargin<5
    tau=[];
end

[n,dimX]=size(X);

if k/n>=0.5
    error('the choice of k is larger than half of the sample size')
end

M=size(u,1);
norm_x=sqrt(sum(X.^2,2));
norm_x_sort=sort(norm_x);
gamma_est=GammaEst(norm_x_sort,k,'Hill');
dq_emp=min(quantile(X*u',k/n),0);
dq=(k/(n*p))^gamma_est*dq_emp';

if dimX<=3
    V=lcon2vert(-u,-dq);
    hest=max(V*u')';
else
    V=qlcon2vert(zeros(dimX,1),-u,-dq);
    hest=nan(M,1);
    for i=1:M
        hest(i)=max(V*u(i,:)');
    end
end


if ~isempty(tau)
    tau=tau(:);
    U_emp=quantile(norm_x,1-k/n);
    ifexd=(X*u'<(ones(n,1)*dq_emp))+0;
    covmat=ifexd'*ifexd/k;
    ifexd_norm=norm_x>U_emp;
    cov_Gam_E=ifexd'*ifexd_norm/k;
    for i=1:numel(cov_Gam_E)
        if cov_Gam_E(i)>0
            cov_Gam_E(i)= cov_Gam_E(i)*(nanmean(log(norm_x(ifexd_norm.*ifexd(:,i)>0)/U_emp))/gamma_est-1); % covariance vector
        end
    end
    covmat=[covmat,cov_Gam_E;cov_Gam_E',1];
    [eigV,D] = eig(covmat);
    lambda=diag(D);
    if min(lambda>0)==0
        lambda(lambda<=0)=10^(-20);
        covmat=eigV*diag(lambda)*eigV';
        [~,covmat]=cov2corr(covmat);
    end
    tempZ=mvnrnd(zeros(size(covmat,1),1),covmat,min(max(3000,size(covmat,1)*10),10^4)); % [E,Gamma]
    delta_approx=tempZ(:,end)/sqrt(k)*log(k/(p*n))*ones(1,M)+tempZ(:,1:end-1)/sqrt(k);
    delta_approx=delta_approx.*(1+tempZ(:,end)/sqrt(k)*ones(1,M));
    cv_1s=quantile(max(delta_approx,[],2),tau)*gamma_est;
    cv_naive=gamma_est*norminv(tau)/sqrt(k)*log(k/(p*n));
    hest_CR=hest*exp(cv_1s');
    hest_naive=hest*exp(cv_naive');
    dq_CR=dq*exp(cv_1s');
    dq_naive=dq*exp(cv_naive');
else
    dq_CR=[];hest_CR=[];
    dq_naive=[];hest_naive=[];
    cv_1s=[];
end

end