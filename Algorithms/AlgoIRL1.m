function [xopt,infos]=AlgoIRL1(x0,F1,F2,w,params)
%------------------------------------------------
% function [xopt,infos]=AlgoIRL1(x0,F1,F2,w,params)
%
% Minimizes the functional
%				       F1(x) + w*F2(G(x))
% for x \in X using the Iterative Reweighted L1 algorithm (IRL1). Assumption on the functionals are:
%      - F1 is proper l.s.c., convex and lipschitz-differentiable (i.e. has gradient (.grad))
%      - F2 is coordinate-wise non decreasing and concave on G(X)
%             -> has an implementation of the sub(super)gradient
%      - With the current implementation G can be:
%             -> the absolute value function
%             -> a function returning the norm-2 of the rows of x (x matrix)
%
% Note: In [1] F1 is not necessarily differentiable. However, since here FISTA [2] is used as inner 
%       algorithm, F1 must be differentiable. In order to deal with other G, other inner algorithm s
%       have to be used.
% 
% Inputs : x0     -> first guess
%          F1     -> Functional F1
%          F2     -> Functional F2
%          w      -> weight parameter (real)
%          params -> Algorithm parameters: structure with fields
%                         .G         -> choice of the functional G:
%                                        - 1: absolute value (default)
%                                        - 2: norm2 of the rows of the variable
%                         .paramsFBS -> structure with parameters for the inner FISTA algorithm (xTol,FTol,maxiter,gam)
%                                       (default : xTol 5e-4, FTol 5e-4, maxiter 10000, gam computed using the lipschitz constant of the gradient of
%                                       F1, if provided otherwise the user has to give this gamma value)
%                         .maxiter   -> maximal number of iterates (default 500)
%                         .xTol      -> stopping criteria on the relative x difference (default 1e-3)
%                         .FTol      -> stopping criteria on the relative F difference (default 1e-3)
%                         .cmptCF    -> if 1 then evaluation of the cost function at each iteration (default 1)
%                                       note: if 0, the stopping criteria is defined only with parameters .xTol and .maxiter
%                         .gpu       -> if 1 then run the algorithm on GPU (default 0)
%                         .verbose   -> if 1 then print main steps (default 0)
%                         .pathSaveTemp -> if non empty, save intermediate estimates during the computation at the given location  
%
% Output : xopt    -> Result of the optimization
%          infos   -> information about the algorithm (structure)
%                         .time     -> elapsed time
%                         .objFun   -> evolution of the objective function (vector)
%                         .stopRule -> stopping rule which has stopped the algorithm
%                                      ('Max Iter' or 'Stationarity')
%                         .nbIter   -> number of iterates performed
%                         .minimizedFunc -> infos about the minimized functional
%                     Note: the parameters w, G, paramsFBS, xTol, FTol and gpu are copied in the info structure
%                           (for reproductibility)
%
% References :
%    [1] Peter Ochs, Alexey Dosovitskiy, Thomas Brox and Thomas Pock, "An iteratively reweighted algorithm for Non-smooth Non-convex
%        optimization in computer vision, SIAM Journal on Imaging Sciences, vol 8, no. 1, pp 331–372 (2015)
%
%    [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
%        SIAM Journal on Imaging Science, vol 2, no. 1, pp 182-202 (2009)
%
% -- Emmanuel Soubies (2016)
%------------------------------------------------

% -- Test if the properties of F1 and F2
if ~isfield(F1,'grad')
	error('In AlgoIRL1: the gradient is not implemented for the functional F1');
end
if ~isfield(F2,'subgrad')
	error('In AlgoIRL1: Functional F2 must have an implementation for the subgradient !');
end

% -- Set default values
if ~isfield(params,'G'), params.G=1; end
if ~isfield(params,'maxiter'), params.maxiter=500; end
if ~isfield(params,'xTol'), params.xTol=1e-3; end
if ~isfield(params,'FTol'), params.FTol=1e-3; end
if ~isfield(params,'cmptCF'), params.cmptCF=1; end
if ~isfield(params,'gpu'), params.gpu=0; end
if ~isfield(params,'verbose'), params.verbose=0; end

% -- Initialization of the function G
if params.G==1
	G=@(x) abs(x);
elseif params.G==2
	G=@(x) sqrt(sum(abs(x).^2,2));
end

tstart=tic;
% -- Main Loop
it=2;
% Initialization
if (params.gpu)
	xopt=gpuArray(x0);
else
	xopt=x0;
end

if params.cmptCF
	infos.objFun(1)=F1.eval(xopt) + w*F2.eval(G(xopt));
end

if params.verbose, 
	fprintf('===========================================================\n');
	fprintf('====        Iteratively Reweighted L1 Algorithm        ====\n');
	fprintf('===========================================================\n \n');
	fprintf('           -----------------------------------\n');
	fprintf('           |  Iteration   |  Objective Func  |\n'); 
	fprintf('           -----------------------------------\n'); 
end

% -- FBS parameters
if isfield(params,'paramsFBS')
	paramsFBS=params.paramsFBS;
end
paramsFBS.cmptCF=0;
paramsFBS.fista=1;
if ~isfield(paramsFBS,'xTol'), paramsFBS.xTol=5e-4; end
if ~isfield(paramsFBS,'FTol'), paramsFBS.FTol=5e-4; end
if ~isfield(paramsFBS,'maxiter'), paramsFBS.maxiter=10000; end
if ~isfield(paramsFBS,'gam')
	if isfield(F1,'lip')
		paramsFBS.gam=0.99/F1.lip;
	else
		error('In AlgoIRL1: Functional F1 do not have an implementation for the lipschitz constant of the gradient => parameter paramsFBS must have the gam field !');
	end
end


while 1
	% -- Update Fold and xold
	if params.cmptCF
		Fold=infos.objFun(it-1);
	end
	xold=xopt;

	% -- Algorithm iteration
	weights=F2.subgrad(G(xopt));                           % compute the weights
	if params.G==1										   % construct weighted l1 functional
		l1_norm=FuncNorm1([],[],weights*w);					
	elseif params.G==2
		l1_norm=FuncNorm1([],[],weights*w,1);
	end   
	[xopt,infosFBS]=AlgoFBS(xopt,F1,l1_norm,1,paramsFBS);  % FISTA algorithm (FBS)
	if strcmp(infosFBS.stopRule,'Max iter')
		disp('In AlgoIRL1: The inner FISTA has reached the maximal number of iterates');
	end

	% -- Convergence test
	if params.cmptCF % if computation of the cost function at each iterate is activated
		infos.objFun(it)=F1.eval(xopt) + w*F2.eval(G(xopt));
		stop=TestConvergence(it,xopt,xold,infos.objFun(it),Fold,params);
	else
		stop=TestConvergence(it,xopt,xold,[],[],params);
	end
	if stop
		break;
	end

	% -- Displaying if verbose option and save intermediate results if activated
	if (mod(it,round(params.maxiter*0.1))==0)
		if params.verbose 
			if params.cmptCF
				fprintf('           |   %6i     |    %1.4e    |\n',it,infos.objFun(it)); 
			else 
				fprintf('           |   %6i     |    %1.4e    |\n',it,F1.eval(xopt) + w*F2.eval(G(xopt))); 
			end		
		end
		if isfield(params,'pathSaveTemp')   
			save([params.pathSaveTemp,'OptivarTemp'],'xopt');	
			if params.cmptCF
				save([params.pathSaveTemp,'infosTemp'],'infos');	
			end	
		end
	end 

	it=it+1;
end
if params.verbose,  fprintf('           -----------------------------------\n'); end

% -- Set infos fields
infos.time=toc(tstart);
if (it==params.maxiter)
	infos.stopRule='Max iter';
else
	infos.stopRule='Stationarity';
end
infos.G=params.G;
infos.nbIter=it;
infos.xTol=params.xTol;
infos.FTol=params.FTol;
infos.paramsFBS=paramsFBS;
infos.cmptCF=params.cmptCF;
infos.F2weight=w;
infos.minimizedFunc=['F1 : ',F1.name,' / F2 : ',F2.name];
infos.gpu=params.gpu;
infos.name='Iterativery Reweighted L1 algorithm';

% -- Display Log
if params.verbose
	fprintf('\n-----> Infos :\n');
	fprintf('      - Stopping rule: %s \n',infos.stopRule);
	fprintf('      - Elapsed time: %7.2f s\n',infos.time);
	fprintf('      - Last Func Eval: %7.3e \n',F1.eval(xopt) + w*F2.eval(G(xopt)));
	fprintf('      - Number of iterates: %i \n',infos.nbIter);
	fprintf('===========================================================\n');
end

% -- Clear the saved OptivarTemp and infosTemp
if isfield(params,'pathSaveTemp')   
	delete([params.pathSaveTemp,'OptivarTemp.mat']);	
	if params.cmptCF
		delete([params.pathSaveTemp,'infosTemp.mat']);	
	end	
end
end
