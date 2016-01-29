function [xopt,infos]=AlgoGradientDescent(x0,F,params)
%------------------------------------------------
% function [xopt,infos]=AlgoGradientDescent(x0,F,params)
%
% Minimizes the differentiable functional F using a gradient descent algorithm
%
% Inputs : x0     -> first guess
%          F      -> Functional to minimize (see the Functional folder)
%          params -> Algorithm parameters: structure with fields
%                         .step    -> descent step
%                         .maxiter -> maximal number of iterates (default 500)
%                         .xTol    -> stopping criteria on the relative x difference (default 1e-3)
%                         .FTol    -> stopping criteria on the relative F difference (default 1e-3)
%                         .gpu     -> if 1 then run the algorithm on GPU (default 0)
%                         .verbose -> if 1 then print main steps (default 0)
%                         .clearTmp -> if 1 then the tmp directory is cleared at the end of the algorithm (default 1)
%
%          Note: if the Functional F is gradient Lipschitz the step has to be lower than 2/L where
%                L is the Lipschitz constant of the gradient. The optimal choice is 1/L (see [1]).
%
% Output : xopt    -> Result of the optimization
%          infos   -> information about the algorithm (structure)
%                         .time     -> elapsed time
%                         .objFun   -> evolution of the objective function (vector)
%                         .stopRule -> stopping rule which has stopped the algorithm
%                                      ('Max Iter' or 'Stationarity')
%                         .nbIter   -> number of iterates performed
%                         .minimizedFunc -> infos about the minimized functional
%                     Note: the parameters step, xTol and FTol are copied in the info structure
%                           (for reproductibility)
%
% Reference :
%    [1] Nesterov, Yurii. "Introductory lectures on convex programming." Lecture Notes (1998): 119-120.
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Clear the tmp folder (in case that a previous algorithm did not stop correctly)
if ~isfield(params,'clearTmp') || params.clearTmp==1
	pid=feature('getpid');
	delete(['/home/esoubies/Desktop/CodeThese/Optimization/tmp/*',num2str(pid),'*']);
end

% -- Test if the functional is differentiable
if ~isfield(F,'grad')
	error('In AlgoGradientDescent: the gradient is not implemented for the given functional');
end
% -- Test if the parameter step is defined
if ~isfield(params,'step')
	error('In AlgoGradientDescent: the algo parameter params.step is not defined');
end

% -- Set default values
if ~isfield(params,'maxiter'), params.maxiter=500; end
if ~isfield(params,'xTol'), params.xTol=1e-3; end
if ~isfield(params,'FTol'), params.FTol=1e-3; end
if ~isfield(params,'gpu'), params.gpu=0; end
if ~isfield(params,'verbose'), params.verbose=0; end

tstart=tic;
% -- Main Loop
it=2;
 % Initialization
if (params.gpu)
	xopt=gpuArray(x0);
else
	xopt=x0;
end
infos.objFun(1)=F.eval(xopt);

if params.verbose, 
	fprintf('===============================================\n');
	fprintf('====      Gradient Descent Algorithm       ====\n');
	fprintf('===============================================\n \n');
	fprintf('     -----------------------------------\n');
	fprintf('     |  Iteration   |  Objective Func  |\n'); 
	fprintf('     -----------------------------------\n'); 
end

while 1
	% -- Update Fold and xold
	Fold=infos.objFun(it-1);
	xold=xopt;

	% -- Algorithm iteration
	xopt=xopt-params.step*F.grad(xopt);

	% -- Convergence test
	infos.objFun(it)=F.eval(xopt);
	stop=TestConvergence(it,xopt,xold,infos.objFun(it),Fold,params);
	if stop
		break;
	end

	% -- Displaying if verbose option
	if params.verbose
		if (mod(it,round(params.maxiter*0.1))==0)
			fprintf('     |   %6i     |    %1.4e    |\n',it,infos.objFun(it));     
		end 
	end

	it=it+1;
end
if params.verbose,  fprintf('     -----------------------------------\n'); end

% -- Set infos fields
infos.time=toc(tstart);
if (it==params.maxiter)
	infos.stopRule='Max iter';
else
	infos.stopRule='Stationarity';
end
infos.nbIter=it;
infos.step=params.step;
infos.xTol=params.xTol;
infos.FTol=params.FTol;
infos.minimizedFunc=F.name;
infos.gpu=params.gpu;
infos.name='Gradient Descent algorithm';

% -- Display Log
if params.verbose
	fprintf('\n-----> Infos :\n');
	fprintf('      - Stopping rule: %s \n',infos.stopRule);
	fprintf('      - Elapsed time: %7.2f s\n',infos.time);
	fprintf('      - Last Func Eval: %7.3e \n',infos.objFun(end));
	fprintf('      - Number of iterates: %i \n',infos.nbIter);
	fprintf('===============================================\n');
end

% -- Clear the tmp folder
if ~isfield(params,'clearTmp') || params.clearTmp==1
	pid=feature('getpid');
	delete(['/home/esoubies/Desktop/CodeThese/Optimization/tmp/*',num2str(pid),'*']);
end
end
