function [gamma,asymstd,stdgamma,thres] = GammaEst(X,k,type)
% Estimation of Extreme Value Index
    tailData=log(X(end-k:end,:));
    thres=exp(tailData(1,:));
    if nargin<3
        % default is Hill Estimator
        type='Hill';
    end
    switch type
        case 'Moment'
            % Moment Estimator by Dekkers, Einmahl and de Haan
            M_1=mean(tailData(2:end,:)-ones(k,1)*tailData(1,:));
            M_2=mean((tailData(2:end,:)-ones(k,1)*tailData(1,:)).^2);
            gamma=M_1+1-1/2*((1-M_1.^2./M_2).^(-1));
            asymstd=sqrt(gamma.^2+1)/sqrt(k);
            stdgamma=std(tailData(2:end,:)-ones(k,1)*tailData(1,:));
        case 'Hill'
            % Hill Estimator
            gamma=mean(tailData(2:end,:)-ones(k,1)*tailData(1,:));
            asymstd=gamma/sqrt(k);
            stdgamma=std(tailData(2:end,:)-ones(k,1)*tailData(1,:),1);
        case 'HillDep'
            % Hill Esimator for dependent data
            % X has only 1 column
            % X is positive?
            if size(X,2)==1
            X_sort=sort(X);
            gamma=mean(log(X_sort(end-k+1:end,:))-ones(k,1)*log(X_sort(end-k,:)));
            stdgamma=std(log(X_sort(end-k+1:end,:))-ones(k,1)*log(X_sort(end-k,:)));
            chi=2*gamma^2*(1/k)*sum(log(max(X(1:end-1,:)/(X_sort(end-k)),1))...
                .*log(max(X(2:end,:)/(X_sort(end-k)),1)))
            phi=gamma*(1/k)*sum(log(max(X(1:end-1,:)/(X_sort(end-k)),1)).*(X(2:end)>X_sort(end-k))...
                +log(max(X(2:end,:)/(X_sort(end-k)),1)).*(X(1:end-1)>X_sort(end-k)));
            omega=2*(1/k)*sum((X(1:end-1)>X_sort(end-k)).*(X(2:end)>X_sort(end-k)));
            asymstd=(1/sqrt(k))*gamma*sqrt(1+chi+omega-2*phi);
            thres=X_sort(end-k);
            else
                xdim=size(X,2);
                gamma=zeros(1,xdim)*nan;
                asymstd=gamma;
                stdgamma=gamma;
                thres=gamma;
                for i=1:xdim
                    [gamma(i),asymstd(i),stdgamma(i),thres(i)]=GammaEst(X(:,i),k,'HillDep');
                end
            end
        case 'MLE'
            nllh=@(theta)(-sum(log(gevpdf(X(end-k+1:end)-X(end-k),theta(1),theta(2),0))));
           options = optimset('fmincon');options = optimset(options,'Display', 'off');options = optimset(options,'Algorithm', 'interior-point');
            mle_para = fmincon(nllh,[GammaEst(X,k,'Hill'),1],[],[],[],[],[-1/2,0],[2,10],[],options);
            gamma=mle_para(1);
            asymstd=(1+gamma)/sqrt(k);
        case 'Bias'
            Z=(tailData(end:-1:end-k+1,:)-tailData(end-1:-1:end-k,:)).*((1:k)');
            w=(1:k)'/(k+1);
            nllh=@(theta)(L2Norm(Z-theta(1)-theta(2)*w.^(theta(3))));
            options = optimset('fmincon');options = optimset(options,'Display', 'off');options = optimset(options,'Algorithm', 'interior-point');
            mle_para = fmincon(nllh,[mean(Z),0,0],[],[],[],[],[0.1,-Inf,0.5],[0.5,Inf,10],[],options);
            gamma=mle_para(1);
        case 'HillSmooth'
            gammavec=zeros(k,1)*nan;
            for i=1:k
                gammavec(i)=GammaEst(X,i,'Hill');
            end
            b=regress(gammavec,[ones(k,1) (1:k)']);
            gamma=b(1);
        case 'Pickands'
            gamma=log((X(end-k+1,:)-X(end-2*k+1,:))./(X(end-2*k+1,:)-X(end-4*k+1,:)))/log(2);
        case 'Zipf'
            n=size(X,1);
            gammavec=zeros(k,1)*nan;
            for i=1:k
                gammavec(i)=GammaEst(X,i,'Hill');
            end
            Z=tailData(end-1:-1:end-k,:).*gammavec;
            j=(1:k)';
            b=regress(Z,[ones(k,1) log((n+1)./(j+1))]);
            gamma=b(2);
        otherwise
            error('Unknown Type of Estimation for EVI')
    end
end
