function [F]=FuncMixNorm2_1(A)
%------------------------------------------------
% function [F]=FuncMixNorm2_1(A)
%
% Constructs the mixed norm ||Ax||_{2,1}  where A is a 
% given operator
%
% Inputs : A -> Operator (see the Operators folder, default : Identity)
%
% Output : F -> Mixed (2,1) Norm: structure with fields
%                 .eval      -> F.eval(x) evaluates the functional at x
%                 .prox      -> F.prox(gam,x) evaluates the proximity operator at x (if it 
%                               is implemented for the given operator)  
%                 .proxFench -> F.proxFench(gam,x) evaluates the proximity operator of the
%                               Fenchel Transform at x (if it is implemented for the given operator)                                 
%                 .name      -> name of the functional
%
% Note : -> when A is the gradient operator, the proximity operator is computed thanks to
%           the Chambolle Pock algorithm
%        -> when A is the 1D gradient operator => use the function FuncNorm1 instead
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- If no operator is given, the Identity is setted
if isempty(A)
	A=OperatorIdentity();
end

% ===== Function definition =====
F.eval = @(x) evaluate(x);

% ===== Prox definition =====
% -- If the given operator is the gradient operator
%    we use the Chambolle Pock algorithm
if strcmp(A.name,'Operator Gradient')
	% -- Gradient Operator (see below the nested function proxOpGrad)
	F.prox = @(gam,y) proxOpGrad(gam,y);
end

% ===== Prox Fenchel definition =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	F.proxFench = @(gam,y)  proxFenOpId(gam,y);
end

% ===== Name =====
F.name='Functional Norm 2-1';

% ===== Nested functions =====
% -- Evaluate function
function y=evaluate(x)
	u=A.eval(x);
	if length(size(u))==3
		% 2D variable
		y=sum(reshape(sqrt(sum(u.^2,3)),numel(u)/2,1));
	elseif length(size(u))==4
		% 3D variable
		y=sum(reshape(sqrt(sum(u.^2,4)),numel(u)/3,1));
	end
end
% -- proximity operator when the gradient operator is used
function x=proxOpGrad(gam,y)
	G=FuncLeastSquares([],y);  % Least squares functional
	F=FuncMixNorm2_1([]);      % Norm 2-1 functional
	nK=(A.norm)^2;             % Norm of the gradient operator
	params.tau=0.05;           % algorithm parameters
	params.sig=1/(nK*params.tau)*0.99;
	params.verbose=0;
	params.maxiter=10000;
	params.xTol=1e-3;
	params.FTol=1e-3;
	[x,~]=AlgoChambollePock(y,F,A,gam,G,params);
end
% -- proximity operator of the Fenchel transform when the identity operator is used
function x=proxFenOpId(gam,y)
	if length(size(y))==3
		nor=repmat(sqrt(sum(y.^2,3)),[1,1,size(y,3)]);
	elseif length(size(y))==4
		nor=repmat(sqrt(sum(y.^2,4)),[1,1,1,size(y,4)]);
	end
	x=y.*(nor <=1) + y./nor.*(nor>1);
end
end
