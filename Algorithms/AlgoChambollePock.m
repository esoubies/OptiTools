function [xopt,infos]=AlgoChambollePock(x0,F,K,w,G,params)
%------------------------------------------------
% function [xopt,infos]=AlgoChambollePock(x0,F,K,w,G,params)
%
% Minimizes the functional
%				       F(Kx) + w*G(x)
% using the Chambolle-Pock algorithm [1]. F and G are two functionals where F has an implementation
% for the proximity operator of its Fenchel transform (.proxFench) and G has an implementation
% for the proximity operator (.prox). K is a linear operator with an implementation of the adjoint.
% 
% Inputs : x0     -> first guess
%          F      -> Functional F
%          K      -> Operator K
%          w      -> weight parameter (real)
%          G      -> Functional G
%          params -> Algorithm parameters: structure with fields
%                         .sig               |
%                         .tau               |-> parameters of the algorithm
%                         .theta (default 1) |
%                         .gam          -> if setted, then the accelerated version of the algorithm (when G of F^* is uniformly
%                                          convex => Grad(G^*) or Grad(F) is 1/gam-Lipschitz) is used (see [1]). 
%                                          If G is uniformly convex then set the parameter var to 1
%                                          If F^* is uniformly convex then set the parameter var to 2
%                         .var          -> select the "bar" variable (see [1]):
%                                             - if 1 then the primal variable xbar = x_n + theta(x_n - x_{n-1}) is used (default)
%                                             - if 2 then the dual variable ybar = y_n + theta(y_n - y_{n-1}) is used
%                         .maxiter      -> maximal number of iterates (default 500)
%                         .xTol         -> stopping criteria on the relative x difference (default 1e-3)
%                         .FTol         -> stopping criteria on the relative F difference (default 1e-3)
%                         .cmptCF       -> if 1 then evaluation of the cost function at each iteration (default 1)
%                                          note: if 0, the stopping criteria is defined only with parameters .xTol and .maxiter
%                         .gpu          -> if 1 then run the algorithm on GPU (default 0)
%                         .verbose      -> if 1 then print main steps (default 0)
%                         .pathSaveTemp -> if non empty, save intermediate estimates during the computation at the given location
%
%          Note: 1- when theta=1, parameters sig and tau have to verrify
%                         sig*tau*||K||^2 < 1
%                   (where ||K|| denotes the norm of the linear operator K) to ensure convergence (see [1]).
%                2- when the accelerated version is used (ie params.gam is setted), sig and tau will be updated at each iteration
%                   and the initial ones (given in params) have to verify
%                         sig*tau*||K||^2 = 1
%
% Output : xopt    -> Result of the optimization
%          infos   -> information about the algorithm (structure)
%                         .time     -> elapsed time
%                         .objFun   -> evolution of the objective function (vector)
%                         .stopRule -> stopping rule which has stopped the algorithm
%                                      ('Max Iter' or 'Stationarity')
%                         .nbIter   -> number of iterates performed
%                         .minimizedFunc -> infos about the minimized functional
%                     Note: the parameters w, sig, tau, theta, gam, var, xTol, FTol and gpu are copied in the info structure
%                           (for reproductibility)
%
% References :
%    [1] Chambolle, Antonin, and Thomas Pock. "A first-order primal-dual algorithm for convex problems with 
%        applications to imaging." Journal of Mathematical Imaging and Vision 40.1, pp 120-145 (2011).
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Test the properties of the functions F, G and the operator K
if ~isfield(F,'proxFench')
	error('In AlgoChambollePock: Functional F must have an implementation for the proximity operator of the Fenchel Transform !');
end
if ~isfield(G,'prox')
	error('In AlgoChambollePock: Functional G must have an implementation for the proximity operator !');
end
if ~isfield(K,'adj')
	error('In AlgoChambollePock: Operator K must be linear and have an implementation of its adjoint !');
end

% -- Test if parameters sig and tau are setted
if ~isfield(params,'sig'), error('In AlgoChambollePock: parameters sig is not setted !'); end
if ~isfield(params,'tau'), error('In AlgoChambollePock: parameters tau is not setted !'); end

% -- Set default values
if ~isfield(params,'var'), params.var=1; end
if ~isfield(params,'theta'), params.theta=1; end
if ~isfield(params,'maxiter'), params.maxiter=500; end
if ~isfield(params,'xTol'), params.xTol=1e-3; end
if ~isfield(params,'FTol'), params.FTol=1e-3; end
if ~isfield(params,'cmptCF'), params.cmptCF=1; end
if ~isfield(params,'gpu'), params.gpu=0; end
if ~isfield(params,'verbose'), params.verbose=0; end

% -- If the norm of K is computed and theta=1, test the condition sig*tau*||K||^2 < 1
if isfield(K,'normA') && params.theta==1
	if params.tau*params.sig*K.normA^2 >=1
		error('In AlgoChambollePock: parameters sig and tau do not satisfy sig*tau*||K||^2 < 1 !');
	end
	if isfield(params,'gam') && (params.tau*params.sig*K.normA^2 ~= 1)
		error('In AlgoChambollePock: parameters sig and tau do not satisfy sig*tau*||K||^2 = 1 as required when using the accelerated version (params.gam is setted) !');
	end
end

% -- Initialize parameters (they can change during the iterations if using the accelerated version)
tau=params.tau;
sig=params.sig;
theta=params.theta;
if isfield(params,'gam')
	gam=params.gam;
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
y=K.eval(xopt);
if params.var==1
	xbar=xopt;
	Kxbar=y;
else
	ybar=y;
	KTybar=K.adj(ybar);
	KTy=KTybar;
end
Kxopt=y;
if params.cmptCF
	infos.objFun(1)=F.eval(Kxopt) + w*G.eval(xopt);
end

if params.verbose, 
	fprintf('===============================================\n');
	fprintf('====       Chambolle-Pock Algorithm        ====\n');
	fprintf('===============================================\n \n');
	fprintf('     -----------------------------------\n');
	fprintf('     |  Iteration   |  Objective Func  |\n'); 
	fprintf('     -----------------------------------\n'); 
end

while 1
	% -- Update Fold and xold
	if params.cmptCF
		Fold=infos.objFun(it-1);
	end
	xold=xopt;
	if params.var==1 % === using xbar
		Kxold=Kxopt;

		% -- Algorithm iteration
		y=F.proxFench(sig,y+sig*Kxbar);
		xopt=G.prox(w*tau,xopt-tau*K.adj(y));
		Kxopt=K.eval(xopt);
		if isfield(params,'gam') % acceleration => uodate theta, tau and sig according to [1]
			theta=1/sqrt(1+2*gam*tau);
			tau=theta*tau;			 
			sig=sig/theta;           
		end
		xbar=(1+theta)*xopt - theta*xold;    
		Kxbar=(1+theta)*Kxopt - theta*Kxold;
	else % === using ybar
		yold=y;
		KTyold=KTy;

		% -- Algorithm iteration
		xopt=G.prox(w*tau,xopt-tau*KTybar);
		Kxopt=K.eval(xopt);
		y=F.proxFench(sig,y+sig*Kxopt);
		KTy=K.adj(y);
		if isfield(params,'gam') % acceleration => uodate theta, tau and sig according to [1]
			theta=1/sqrt(1+2*gam*sig); 
			tau=tau/theta;		 
			sig=sig*theta;        
		end
		ybar=(1+theta)*y - theta*yold;    
		KTybar=(1+theta)*KTy - theta*KTyold;
	end

	% -- Convergence test
	if params.cmptCF % if computation of the cost function at each iterate is activated
		infos.objFun(it)=F.eval(Kxopt) + w*G.eval(xopt);
		stop=TestConvergence(it,xopt,xold,infos.objFun(it),Fold,params);
	else
		stop=TestConvergence(it,xopt,xold,[],[],params);
	end
	% -- Break if stopping criteria reached
	if stop && it>2 % condition it>2 is to avoid to stop the algorithm if the first iteration don't move xopt (it can be the case by initializing with 0)
		break;
	end

	% -- Displaying if verbose option and save intermediate results if activated
	if (mod(it,round(params.maxiter*0.1))==0)
		if params.verbose 
			if params.cmptCF
				fprintf('     |   %6i     |    %1.4e    |\n',it,infos.objFun(it)); 
			else 
				fprintf('     |   %6i     |    %1.4e    |\n',it,F.eval(Kxopt) + w*G.eval(xopt)); 
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
if params.verbose,  fprintf('     -----------------------------------\n'); end

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
infos.cmptCF=params.cmptCF;
infos.sig=params.sig;
infos.tau=params.tau;
infos.theta=params.theta;
if isfield(params,'gam')
	infos.gam=gam;
end
infos.Gweight=w;
infos.minimizedFunc=['F : ',F.name,' / K : ',K.name,' / G : ',G.name];
infos.var=params.var;
infos.gpu=params.gpu;
infos.name='Chambolle Pock algorithm';

% -- Display Log
if params.verbose
	fprintf('\n-----> Infos :\n');
	fprintf('      - Stopping rule: %s \n',infos.stopRule);
	fprintf('      - Elapsed time: %7.2f s\n',infos.time);
	fprintf('      - Last Func Eval: %7.3e \n',F.eval(Kxopt) + w*G.eval(xopt));
	fprintf('      - Number of iterates: %i \n',infos.nbIter);
	fprintf('===============================================\n');
end

% -- Clear the saved OptivarTemp
if isfield(params,'pathSaveTemp')   
	delete([params.pathSaveTemp,'OptivarTemp.mat']);	
	if params.cmptCF
		delete([params.pathSaveTemp,'infosTemp.mat']);	
	end				
end
end
