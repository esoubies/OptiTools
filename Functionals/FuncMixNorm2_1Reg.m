function [F]=FuncMixNorm2_1Reg(A,epsl,varargin)
%------------------------------------------------
% function [F]=FuncMixNorm2_1Reg(A,epsl,varargin)
%
% Constructs the mixed norm ||Ax||_{2,1,epsl}  where A is a 
% given operator and epsl a real positive used to get the 
% differenciability at 0.
%
% Inputs : A    -> Operator (see the Operators folder, default : Identity)
%          epsl -> regularization parameter (>0) to have the differenciability at 0 (default 1e-6)
%          ind  -> (first varargin parameter) structure with fields:
%                 .i     |   Sub indices of the variable x used to evaluate the functional only on these 
%                 .j     |-> indices. i, j and k correspond to the 3 dimensions. all the fields have to
%                 .k     |   be setted.
%                            Example : you have an optimization variable defined by x=[f,b] and you           
%                                      want put a TV regularizer only on the part f of x :
%                                               G=OperatorGradient(1);
%                                               ind.i=1:length(f);ind.j=1;ind.k=1;
%                                               F=FuncNorm1(G,ind); 
%
% Output : F -> Mixed (2,1) Regularized Norm: structure with fields
%                 .eval   -> F.eval(x) evaluates the functional at x
%                 .grad   -> F.grad(x) evaluates the gradient at x (if the 
%                            gradient is implemented for the given operator)                             
%                 .name   -> name of the functional
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Get the optional inputs (varargin)
if nargin > 2
	ind=varargin{1};
	if ~isfield(ind,'i') || ~isfield(ind,'j')  || ~isfield(ind,'k') 
		error('In FuncMixNorm2_1Reg : optional parameter ind must contain fields i. .j and .k');
	end
else
	ind=[];
end
if isempty(epsl)
	epsl=1e-6;
end

% -- If no operator is given, the Identity is setted
if isempty(A)
	A=OperatorIdentity();
end

% ===== Function definition =====
F.eval = @(x) evaluate(x);

% ===== Gradient definition =====
F.grad = @(x) grad(x);

% ===== Name =====
F.name=['Functional Norm 2-1 Regularized combined with ',A.name];

% ===== Nested functions =====
% -- Evaluate function
function y=evaluate(x)
	if isempty(ind)
		u=A.eval(x);
	else
		u=A.eval(x(ind.i,ind.j,ind.k));
	end
	if length(size(u))==2
		% 1D variable
		y=sum(sqrt(u.^2 + epsl));
	elseif length(size(u))==3
		% 2D variable
		y=sum(reshape(sqrt(sum(u.^2,3) + epsl),numel(u)/2,1));
	elseif length(size(u))==4
		% 3D variable
		y=sum(reshape(sqrt(sum(u.^2,4)+epsl),numel(u)/3,1));
	end
end
% -- Gradient of the functional
function g=grad(x)
	if isempty(ind)
		u=A.eval(x);
	else
		u=A.eval(x(ind.i,ind.j,ind.k));
	end
	if length(size(u))==2
		nor=sqrt(u.^2+epsl);
	elseif length(size(u))==3
		nor=repmat(sqrt(sum(u.^2,3)+epsl),[1,1,size(u,3)]);
	elseif length(size(u))==4
		nor=repmat(sqrt(sum(u.^2,4)+epsl),[1,1,1,size(u,4)]);
	end
	if isempty(ind)
		g=A.adj(u./nor);
	else
		g=x*0;
		g(ind.i,ind.j,ind.k)=A.adj(u./nor);
	end
end
end
