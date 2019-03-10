function [cov_prob_refine,cov_prob] = simfun(para)
rng('default')
htrue=cell(para.nrdst,numel(para.p));
para.u2D=dirrnd(2,para.m); % generate directions
para.u6D=dirrnd(6,para.m); % generate directions

% calculate true values of support functions
for i=1:numel(para.p)
    % bivariate Cauchy
    htrue{1,i}=tinv(1-para.p(i),1)*ones(para.m*2,1);
    % bivariate t3
    htrue{2,i}=tinv(1-para.p(i),3)*ones(para.m*2,1);
    % bivariate affine t3
    htrue{3,i}=sqrt(sum((para.u2D*sqrtm(para.Sigma)).^2,2))*tinv(1-para.p(i),3);
    % six-dimensional affine t-3.5
    htrue{4,i}=sqrt(sum((para.u6D*sqrtm(para.Sigma6D)).^2,2))*tinv(1-para.p(i),3.5);
end
%% Coverage
ifcov=nan(para.nrdst,numel(para.tau),numel(para.n),para.S);
ifcov_refine=nan(para.nrdst,numel(para.tau),numel(para.n),para.S);
rng('default') % for replication
for i=1:para.S
    %i
    for w=1:numel(para.p)
        for j=1:para.nrdst
            %j
            switch j
                % generate observations
                case 1
                    X=mvtrnd(eye(2),1,para.n(w));
                    u=para.u2D;
                case 2
                    X=mvtrnd(eye(2),3,para.n(w));
                    u=para.u2D;
                case 3
                    X=mvtrnd(eye(2),3,para.n(w))*sqrtm(para.Sigma);
                    u=para.u2D;
                otherwise
                    X=mvtrnd(eye(6),3.5,para.n(w))*sqrtm(para.Sigma6D);
                    u=para.u6D;
            end
            % estimate quantile set
            [~,hest_CR,hest_naive]=Q_hat(X,para.p(w),para.k(w,j),u,para.tau);
            for t=1:numel(para.tau)
                ifcov(j,t,w,i)=min(log(htrue{j,w})<=log(hest_naive(:,t)));
                ifcov_refine(j,t,w,i)=min(log(htrue{j,w})<=log(hest_CR(:,t)));
            end
        end
    end
end

cov_prob=nanmean(ifcov,4); % empirical coverage probability of first-order confidence regions
cov_prob_refine=nanmean(ifcov_refine,4); % empirical coverage probability of second-order confidence regions
end

