%--------------------------------------------------
% TV-Denoising Example
%		This script presents an exemple of denoising 
% using a TV regularized least squares minimization.
% The optimization is performed with the Chambolle-
% Pock algorithm (accelerated version)
%
% Emmanuel Soubies (2015)
%--------------------------------------------------
clear all; close all; clc;

% === Dimension of the example (2D : 2 / 3D : 3)
dim=2;
gpu=0; % to perform the algorithm on GPU (requires the Parallel Computing Toolbox)

% === Noise parameters
s=0.1; % standard deviation of the gaussian noise

% === Algorithm parameter
w=10; % hyperparameter
params.tau=0.05; % sig will be computed to satisfy the convergence inequality
params.verbose=1;
params.maxiter=500;
params.xTol=1e-5;
params.FTol=1e-5;
params.gpu=gpu;

% === Data reading
if dim==2
	im=double(imread('cameraman.tif'));im=im/max(im(:));
	figure; imagesc(im); axis image; axis off; colormap gray;
	title('Original image');
elseif dim==3
	load mri;
	D=reshape(D,size(D,1),size(D,2),size(D,4));
	im=double(D(:,:,:,1)); im=im/max(im(:));
	implay(im);
end

% === Add noise
randn('seed',1);
imb=im + s*randn(size(im)); 
if dim==2
	figure; imagesc(imb); axis image; axis off; colormap gray;
	title('Noisy image');
elseif dim==3
	implay(imb);
end

% - if GPU then convert the image as a gpuArray
if gpu
	imb=gpuArray(imb);
end

% === Solver
% - TV regularization
if dim==2
	G=OperatorGradient(2);
elseif dim==3
	G=OperatorGradient(3);
end
F1=FuncMixNorm2_1([]);
nG=(G.norm)^2;
params.sig=1/(nG*params.tau);

% - Least-Squares functional
F2=FuncLeastSquares([],imb);
params.gam=0.5*w/F2.lipFench;       % to use the accelerated version of the algorithm since F2 is uniformly convex

% - Chambolle-Pock algorithm
[im_debruit,infos]=AlgoChambollePock(imb,F1,G,w,F2,params);

% Denoised image
if dim==2
	figure; imagesc(im_debruit); axis image; axis off; colormap gray;
	title('Denoised image');
elseif dim==3
	if gpu
		implay(gather(im_debruit));
	else
		implay(im_debruit);
	end
end

% Objective Function evolution
figure;
plot(1:length(infos.objFun),infos.objFun);
xlabel('Iterations');
ylabel('Cost Function');
title('Cost Function Evolution');
set(gca,'FontSize',12);grid;

% Log Infos
disp('Log informations:');
infos
