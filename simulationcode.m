%% Simulations
% Yi He: 31/01/2019
% Initial Parameters
para.nrdst=4; % number of distributions
para.dstname={'Biv. Cauchy','Biv. t_3','Affine t_3','Six-dimensional t_{3.5}'}; % name of distributions
para.epsname={'Cauchy-2D','t3-2D','t3-2D-Affine','t-6D'};
para.n=[1500,5000]; % sample sizes
para.Sigma=[2,0.3;0.3,1];
load datafit emp_cov % load the calibrated covariance matrix from the real-life data set
para.Sigma6D=emp_cov*(3.5-2)/3.5;
clear emp_cov
para.m=500; % number of direction per dimension
para.S=1000; % number of scenarios
para.tau=[0.9,0.95,0.99]; % confidence level
para.p=1./para.n; % probability level
para.kinitial=[150,40,80,100;400,100,180,200]; %choices of k's 
%%
para.k=para.kinitial;
[cov_prob_refine,cov_prob] = simfun(para);
para.k=1.5*para.kinitial;
[cov_prob_refine_largek,cov_prob_largek] = simfun(para);
para.k=0.5*para.kinitial;
[cov_prob_refine_smallk,cov_prob_smallk] = simfun(para);
%%
%save simulationdata para cov_prob_refine cov_prob cov_prob_refine_largek cov_prob_largek cov_prob_refine_smallk cov_prob_smallk
