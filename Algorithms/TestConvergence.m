function stop=TestConvergence(it,x,xold,Fnew,Fold,params);
%------------------------------------------------
% function stop=TestConvergence(it,x,xold,Fnew,Fold,params);
%
% Check if the stopping rule (variable + function stationarity) or (maxiter)
% is reached
%
% Inputs : it     -> current iteration
%          x      -> current x value
%          xold   -> old x value
%          Fnew   -> new objective function value (evaluated at x)
%          Fold   -> old objective function value (evaluated at xold)
%          params -> algorithm parameters (used to know the values xTol FTol and maxiter)
%
% Output : stop -> if 1 the stopping rule is verified
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Relative x difference
xdiff=norm(reshape(x-xold,[],1),'fro')/(norm(xold(:),'fro')+eps);

% -- Relative F difference
if ~isfield(params,'cmptCF') || params.cmptCF % if the computation of the cost function is activated 
	Fdiff=abs(Fnew-Fold)/(abs(Fold)+eps);   % variable + function stationarity
	stationarity=(xdiff<params.xTol) && (Fdiff<params.FTol);
else                                        % else only variable stationarity
	stationarity=(xdiff<params.xTol);
end

% -- Check if the stopping rule is reached
stop=((stationarity)||(it>=params.maxiter));
end
