%% Programmed by Yi He (yi.he2@monash.edu)
% Matlab version: R2017b
%% Import Data
clc
clear
linebreak='------------------------------------------------------';
load mktdata countrycode rt dates
countryname={'Germany','France','UK','Italy','Japan','USA'};
disp('Loading stock market data...')
%% GARCH Calibration: global self weigthed exp likelihood estimator (Zhu and Ling, 2011, AOS)
[T,N]=size(rt); % number of periods and number of countries
% Fit AR(1)-GARCH(1,1) model for each time series
disp('Fitting AR(1)-GARCH(1,1) model for each time series...')
mdl=cell(N,1);
for i=1:N
    mdl{i}=weightedGARCH(rt(:,i));
end

%% Calculate univariate residuals
disp('Calculating residual returns...')
res=nan(T,N);varts=res;rt_med=nan(1,N);
for i=1:N
    rt_med(i)=mdl{i}.Offset;
    varts(~isnan(rt(:,i)),i)=infer(mdl{i},rt(:,i));
    res(:,i)=(rt(:,i)-mdl{i}.Offset)./sqrt(varts(:,i));
end
%% Export the sample covariance matrix to be used in simulation
emp_cov=nancov(res);
%save datafit emp_cov

%% Ljung-Box tests for nomial and squared residuals
lbqtesth=zeros(2,N);
for i=1:N
    lbqtesth(1,i)=lbqtest(res(:,i));
    lbqtesth(2,i)=lbqtest(res(:,i).^2);
end
disp(linebreak);
disp('The outcome of Ljung-Box tests for residuals:')
testoutcome={'do not reject', 'reject'};
for i=1:N
    disp([countryname{i},'--',testoutcome{lbqtesth(1,i)+1}]);
end
disp(linebreak);
disp('The outcome of Ljung-Box tests for squared residuals:')
testoutcome={'do not reject', 'reject'};
for i=1:N
    disp([countryname{i},'--',testoutcome{lbqtesth(2,i)+1}]);
end
%% Testing Tail Equivalence
disp(linebreak);
rng('default')
k_test=90;
[h,pval] = testtaileq([-res res],k_test);
disp(['The tail indices in order (k=',num2str(k_test),'):'])
tempGam=round(sort(nanGammaEst(sort([-res,res]),k_test,'Hill')),2);
tempstr=num2str(tempGam(1));
for i=2:numel(tempGam)
    tempstr=[tempstr,',',num2str(tempGam(i))];
end
disp(tempstr);
disp(['Minmax Tail Shape test p-value:',num2str(pval)]);
clear tempGam tempstr
%% Repeat the test
rng('default')
disp(linebreak);
disp('Repeating the minimax tail shape tests for different k between 60 and 110...');
k_test_all=10:10:100;
pval_all=nan(length(k_test_all),1);
for i=1:length(k_test_all)
    [~,pval_all(i)] = testtaileq([-res res],k_test_all(i));
end
disp(['Minimal p-value:',num2str(min(pval_all))]);
disp(['Average p-value:',num2str(mean(pval_all))]);
disp(['Maximal p-value:',num2str(max(pval_all))]);
disp(['Outcome based on the minimal p-value: ',testoutcome{(min(pval_all)<0.05)+1},' at 95% confidence level']);
%% Hill Plot for residual returns
disp(linebreak)
disp('Making the Hill plots for six-dimensional residuals...')
res_valid=res(~isnan(sum(res,2)),:);
dates_valid=dates(~isnan(sum(res,2)));
res_norm=sort(sqrt(sum(res_valid.^2,2))); % norm of observations
n=numel(res_norm);
kall=1:round(n/2);
gamma_res=nan(numel(kall),1);
for k=kall
        gamma_res(k)=nanmean(log(res_norm(end-k+1:end))-log(res_norm(end-k)));
end
disp('Displaying the plot on screen...')
figure('pos',[50 50 1200 400])
plot(kall,gamma_res)
ylabel('Hill estimate');
xlabel('k');
set(gca,'XTick',0:50:600)
disp('Saving the plot in local folder as an eps file ...')
saveas(gca,['HillPlot_Residual','_n',num2str(n)],'epsc');
%% Quantile Region for residual returns
disp(linebreak);
disp('Estimating Quantile Region at level p=0.01%...');
% Randomly Generate 500*6=3000 Directions on Unit Sphere
rng('default')
u6D=dirrnd(6,500);
tau=0.5:0.01:0.99;
[hest,hest_CR,hest_naive,dq,dq_naive,dq_CR,gamma_est,cv_1s] =Q_hat(res_valid,0.01/100,180,u6D,tau);
%% Detecting outlier
disp('Detecting outliers outside the quantile region...')
outlier_est=dates_valid(max(res_valid*u6D'>(ones(n,1)*hest'),[],2));
outlier_CR=cell(numel(tau),1);
for i=1:numel(tau)
outlier_CR{i}=dates_valid(max(res_valid*u6D'>(ones(n,1)*hest_CR(:,i)'),[],2));
end

%% Output outlier
disp(['Detected ', num2str(numel(outlier_est)), ' outlier(s) on the following dates (p values provided):'])
for i=1:numel(outlier_est)
    j=0;
   while sum(outlier_CR{j+1}==outlier_est(i))>0 && j<=numel(tau)
       j=j+1;
   end
   disp([num2str(outlier_est(i)),': ',num2str(1-tau(j))]);
end
%% Robust Portfolios
[hest,hest_CR,hest_naive,dq,dq_naive,dq_CR,gamma_est,cv_1s] =Q_hat(res_valid,0.01/100,180,u6D,tau);
%%
rt_mean=nanmean(rt);
varts_valid=varts(~isnan(sum(res,2)),:);
rt_valid=rt(~isnan(sum(res,2)),:);

port_tau=0.9;port_tau_ix=nan(numel(port_tau),1);
for i=1:numel(port_tau)
    port_tau_ix(i)=find(tau==port_tau(i));
    if isnan(port_tau_ix(i))
        error('cannot find this confidence level from the estimation')
    end
end
port_weight=nan(n,N,numel(port_tau)+1);
port_rt=nan(n,numel(port_tau)+1);
port_rt_riskydate=false(n,2);
worst_ts=nan(n,numel(port_tau_ix)+1);
worstlevel=-0.1;
for t=1:n
    t
    all_port=-u6D/diag(sqrt(varts_valid(t,:)));
    all_port_worst=[-hest,-hest_CR(:,port_tau_ix)];
    % only consider the loss of the porfolios
    all_port_worst(sum(all_port,2)<=0,:)=[];
    all_port(sum(all_port,2)<=0,:)=[];
    all_port=all_port./(sum(all_port,2)*ones(1,6));
    all_port_worst=all_port_worst.*(sqrt(diag(all_port*diag(varts_valid(t,:))*all_port'))*ones(1,size(all_port_worst,2)))+all_port*rt_med'*ones(1,size(all_port_worst,2));
    all_port_mean=all_port*rt_mean';
    for i=1:size(all_port_worst,2)
        all_port_mean_temp=all_port_mean;
        worst_ts(t,i)=-max(all_port_worst(:,i));
        if sum(all_port_worst(:,i)>=worstlevel)>0
            all_port_mean_temp(all_port_worst(:,i)<worstlevel,:)=-inf;
            [~,port_optix]=max(all_port_mean_temp);
            port_weight(t,:,i)=all_port(port_optix,:);
        else
            port_rt_riskydate(t,i)=true;
            [~,port_optix]=max(all_port_worst(:,i));
            %port_weight(t,:,i)=port_weight(t,:,1);
            port_weight(t,:,i)=all_port(port_optix,:);
        end
        port_rt(t,i)=port_weight(t,:,i)*rt_valid(t,:)';
    end
end
clear all_port all_port_worst all_port_mean all_port_mean_temp
sum(port_rt_riskydate)

 








