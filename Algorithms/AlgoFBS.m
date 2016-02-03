function [xopt,infos]=AlgoFBS(x0,F,G,w,params)
%------------------------------------------------
% function [xopt,infos]=AlgoFBS(x0,F,G,w,params)
%
% Minimizes the functional
%				       F(x) + w*G(x)
% using the Forward-Backward Splitting (FBS) algorithm [1]. F and G are two functionals where  G has 
% an implementation for the proximity operator (.prox) and F is differentiable (i.e. has gradient (.grad))
% 
% Inputs : x0     -> first guess
%          F      -> Functional F
%          G      -> Functional G
%          w      -> weight parameter (real)
%          params -> Algorithm parameters: structure with fields
%                         .gam     -> descent step parameter of the algorithm
%                         .fista   -> if 1 use the accelerated version FISTA [3] (default 0)
%                         .maxiter -> maximal number of iterates (default 500)
%                         .xTol    -> stopping criteria on the relative x difference (default 1e-3)
%                         .FTol    -> stopping criteria on the relative F difference (default 1e-3)
%                         .cmptCF  -> if 1 then evaluation of the cost function at each iteration (default 1)
%                                     note: if 0, the stopping criteria is defined only with parameters .xTol and .maxiter
%                         .gpu     -> if 1 then run the algorithm on GPU (default 0)
%                         .verbose -> if 1 then print main steps (default 0)
%                         .pathSaveTemp -> if non empty, save intermediate estimates during the computation at the given location
%
%          Note: when the functional are convex and F has a Lipschitz continuous gradient, convergence is
%                ensured by taking gam in (0,2/L] where L is the Lipschitz constant of grad(F) (see [1]).
%                When FISTA is used [3], gam should be in (0,1/L] 
%                For nonconvex functions [2] take gam in (0,1/L]       
%
% Output : xopt    -> Result of the optimization
%          infos   -> information about the algorithm (structure)
%                         .time     -> elapsed time
%                         .objFun   -> evolution of the objective function (vector)
%                         .stopRule -> stopping rule which has stopped the algorithm
%                                      ('Max Iter' or 'Stationarity')
%                         .nbIter   -> number of iterates performed
%                         .minimizedFunc -> infos about the minimized functional
%                     Note: the parameters w, gam, fista, xTol, FTol, cmptCF and gpu are copied in the info structure
%                           (for reproductibility)
%
% References :
%    [1] P.L. Combettes and V.R. Wajs, "Signal recovery by proximal forward-backward splitting", SIAM Journal on
%        Multiscale Modeling & Simulation, vol 4, no. 4, pp 1168-1200, (2005).
%
%    [2] Hedy Attouch, Jerome Bolte and Benar Fux Svaiter "Convergence of descent methods for semi-algebraic and 
%        tame problems: proximal algorithms, forward-backward splitting, and regularized gaussiedel methods." 
%        Mathematical Programming, 137 (2013).
%
%    [3] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
%        SIAM Journal on Imaging Science, vol 2, no. 1, pp 182-202 (2009)
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Test if the properties of F and G
if ~isfield(F,'grad')
	error('In AlgoFBS: the gradient is not implemented for the functional F');
end
if ~isfield(G,'prox')
	error('In AlgoFBS: Functional G must have an implementation for the proximity operator !');
end

% -- Test if parameters gam is setted
if ~isfield(params,'gam'), error('In AlgoFBS: parameters gam is not setted !'); end

% -- Set default values
if ~isfield(params,'maxiter'), params.maxiter=500; end
if ~isfield(params,'xTol'), params.xTol=1e-3; end
if ~isfield(params,'FTol'), params.FTol=1e-3; end
if ~isfield(params,'cmptCF'), params.cmptCF=1; end
if ~isfield(params,'fista'), params.fista=0; end
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
if params.cmptCF
	infos.objFun(1)=F.eval(xopt) + w*G.eval(xopt);
end

if params.verbose, 
	fprintf('===========================================================\n');
	fprintf('====       Forward-Backward Splitting Algorithm        ====\n');
	fprintf('===========================================================\n \n');
	fprintf('           -----------------------------------\n');
	fprintf('           |  Iteration   |  Objective Func  |\n'); 
	fprintf('           -----------------------------------\n'); 
end

if params.fista
	tk=1;
	y=xopt;
end
while 1
	% -- Update Fold and xold
	if params.cmptCF
		Fold=infos.objFun(it-1);
	end
	xold=xopt;

	% -- Algorithm iteration
	if params.fista 
		% if fista
		xopt=G.prox(params.gam*w,y - params.gam*F.grad(y));
		told=tk;
		tk=0.5*(1+sqrt(1+4*tk^2));
		y=xopt + (told-1)/tk*(xopt-xold);
	else 
		xopt=G.prox(params.gam*w,xopt - params.gam*F.grad(xopt));
	end

	% -- Convergence test
	if params.cmptCF % if computation of the cost function at each iterate is activated
		infos.objFun(it)=F.eval(xopt) + w*G.eval(xopt);
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
				fprintf('           |   %6i     |    %1.4e    |\n',it,F.eval(xopt) + w*G.eval(xopt)); 
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
infos.nbIter=it;
infos.xTol=params.xTol;
infos.FTol=params.FTol;
infos.gam=params.gam;
infos.cmptCF=params.cmptCF;
infos.fista=params.fista;
infos.Gweight=w;
infos.minimizedFunc=['F : ',F.name,' / G : ',G.name];
infos.gpu=params.gpu;
infos.name='Forward-Backward Splitting algorithm';

% -- Display Log
if params.verbose
	fprintf('\n-----> Infos :\n');
	fprintf('      - Stopping rule: %s \n',infos.stopRule);
	fprintf('      - Elapsed time: %7.2f s\n',infos.time);
	fprintf('      - Last Func Eval: %7.3e \n',F.eval(xopt) + w*G.eval(xopt));
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
