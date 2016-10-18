%-----------------------------------------------------------------
% This script provides an example of how to use the FBS  and IRL1 
% algorithms to solve the CEL0 penalized least squares problem.
%
% -- Emmanuel Soubies (2015)
%-----------------------------------------------------------------
clear all; close all; clc;

% -- Parameters for generation
M=128;   % size of the measured vector
N=256;   % size of the unknown signal
K=35;    % number of non-zero values in the generated signal
sig=0.5;   % noise standard deviation
cmplx=0;   % Complex values
alg='irl1';% Algorithm: 'fbs' -> Forward Backward Splitting 
		   %            'irl1'-> Iteratively Reweighted-L1 algorithm
% - Set the seed-number (for reproductibility)
rand('state', 1);
randn('state', 1);

%%%%%%%%%%%%%%%%%%%%
%% Data generation (Real)
%%%%%%%%%%%%%%%%%%%%
if ~cmplx
    A=randn(M,N);                                   % Mixture matrix
    x=zeros(N,1);                                   % Signal
    sel = randperm(N); sel = sel(1:K);              % Location of the non-zero entries
    x(sel) = sign(randn(K,1)) .* (1-.8*rand(K,1));  % set the non-zeros values randomly such that their amplitude is greater than 0.2
    h=figure; ind=find(abs(x)>eps); stem(ind,x(ind),'x'); hold all; axis([0 N -1.8 1.8]);
    %
    y=A*x + randn(M,1)*sig;                         % Measurements vector
    figure; stem(A*x,'*'); hold all; stem(y,'x','LineStyle','none'); grid; axis([0 M min(min(A*x),min(y)) max(max(A*x),max(y))]);
    legend('Noiseless','Noisy');title('\fontsize{14} \it \bf Observed data');
end

%%%%%%%%%%%%%%%%%%%%
%% Data generation (Complex)
%%%%%%%%%%%%%%%%%%%%
if cmplx
    A=randn(M,N)+1i*randn(M,N);          % Mixture matrix
    x=zeros(N,1);                        % Signal
    sel = randperm(N); sel = sel(1:K);   % Location of the non-zero entries
    x(sel) = randn(K,1) + 1i*randn(K,1);              % set the non-zeros values randomly ... 
    x(sel) = x(sel)./abs(x(sel)) .* (1-.8*rand(K,1)); % ... such that their amplitude is greater than 0.2
    h=figure; ind=find(abs(x)>eps); stem(ind,real(x(ind)),'xb'); hold on;  stem(ind,imag(x(ind)),'xr'); axis([0 N -1.8 1.8]);
    %
    y=A*x + randn(M,1)*sig + 1i*randn(M,1)*sig;       % Measurements vector
    figure; stem(real(A*x),'*b'); hold on; stem(imag(A*x),'*r'); stem(real(y),'xb','LineStyle','none'); stem(imag(y),'xr','LineStyle','none'); grid; axis([0 M -max(max(abs(A*x)),min(abs(y))) max(max(abs(A*x)),max(abs(y)))]);
    legend('Noiseless Real','Noiseless Imag','Noisy Real','Noisy Imag');title('\fontsize{14} \it \bf Observed data');
end
%%%%%%%%%%%%%%%%%
%% Optimization
%%%%%%%%%%%%%%%%%
% -- Parameters
% Functional
if cmplx
    lamb=2;  % lambda parameter of the CEL0 penalty
else
    lamb=1;
end
% Algo
params.verbose=1;       % to display the iterations evolution (otherwise put 0)
params.maxiter=1000;    % maximal number of iterations
params.xTol=1e-5;       % tolerance on the relative difference of x between two successive iterations
params.FTol=1e-5;       % tolerance on the relative difference  cost function value between two successive iterations
params.gpu=0;           % put 1 to run th code on gpu (requires the Parallel Computing Toolbox)
x0=zeros(size(x));      % initialization of the algorithm to 0

% -- Construct the linear operator from the matrix A
Op=OperatorFromMatrix(A);

% -- Construct the Least-Squares functional (Data term)
F=FuncLeastSquares(Op,y);
params.gam=1/F.lip;    % gamma parameter of the FBS algorithm (must be < to 1/L where L=||A^t*A||, it is given by the 
					   % field nAtA of the operator

% -- Construct the CEL0 penalty
P=FuncCEL0(Op.nai,lamb,[]);

% -- Minimization of F + P
if strcmp(alg,'fbs')
	% with the FBS algorithm
	params.gam=0.99/F.lip;  % gamma parameter of the FBS algorithm (must be < to 1/L where L=||A||^2, with ||A|| given by the 
						    % field .norm of the operato
	[xopt,infos]=AlgoFBS(x0,F,P,1,params);
elseif strcmp(alg,'irl1')
	% with the IRL1 algorithm
	params.G=1;
	[xopt,infos]=AlgoIRL1(x0,F,P,1,params);
end

% -- Plots results
if ~cmplx
   	figure(h); ind=find(abs(xopt)>eps); stem(ind,xopt(ind),'o');grid;
   	legend('True signal','Estimated signal');
else
   	figure(h); ind=find(abs(xopt)>eps); stem(ind,real(xopt(ind)),'om','LineStyle','none'); stem(ind,imag(xopt(ind)),'dk','LineStyle','none');grid;
   	legend('True signal (real)','True signal (imag)','Estimated signal (real)','Estimated signal (imag)');
end

% -- Plot cost function
figure;
plot(infos.objFun()); xlabel('Iterations'); ylabel('Cost function'); grid;
% -- Display algorithm output infos
infos
