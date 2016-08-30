function [F]=FuncNorm1(A,varargin)
%------------------------------------------------
% function [F]=FuncNorm1(A,varargin)
%
% Constructs the norm ||Ax||_{1}  where A is a given operator
%
% Inputs : A      -> Operator (see the Operators folder, default : Identity)
%          ind    -> (first varargin parameter) structure with fields:
%                   .i     |   Sub indices of the variable x used to evaluate the functional only on these 
%                   .j     |-> indices. i, j and k correspond to the 3 dimensions. all the fields have to
%                   .k     |   be setted.
%                            Example : you have an optimization variable defined by x=[f,b] and you           
%                                      want put a TV regularizer only on the part f of x :
%                                               G=OperatorGradient(1);
%                                               ind.i=1:length(f);ind.j=1;ind.k=1;
%                                               F=FuncNorm1(G,ind); 
%          w      -> (second varargin parameter) weights of each components of (Ax). The size of w will 
%                    determine the one of the variable x which can be used with the created functional. 
%          struct -> boolean: if true (1) then the row-sparsity is used,  that is apply first a l2-norm on
%                    the row of the matrix variable x and then sum the result.
%
%
% Output : F -> Norm-1 : structure with fields
%                 .eval      -> F.eval(x) evaluates the functional at x
%                 .prox      -> F.prox(gam,x) evaluates the proximity operator at x (if it 
%                               is implemented for the given operator)  
%                 .proxFench -> F.proxFench(gam,x) evaluates the proximity operator of the
%                               Fenchel Transform at x (if it is implemented for the given operator)                       
%                 .name      -> name of the functional
%
% Note : when A is the (1D) gradient operator, the proximity operator is computed thanks to
%        the Chambolle Pock algorithm
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Get the optional inputs (varargin)
if nargin == 2
	ind=varargin{1}; w=1; struct=0;
elseif nargin == 3
	ind=varargin{1}; w=varargin{2}; struct=0;
elseif nargin == 4
	ind=varargin{1}; w=varargin{2}; struct=varargin{3};
end
if ~isempty(ind)
	if ~isfield(ind,'i') || ~isfield(ind,'j')  || ~isfield(ind,'k') 
		error('In FuncNorm1 : optional parameter ind must contain fields i. .j and .k');
	end
end

% -- If no operator is given, the Identity is setted
if isempty(A)
	A=OperatorIdentity();
end

% ===== Function definition =====
if struct % l2-1 
	F.eval = @(x) evaluateStruct(x);
else	  % classic l1
	F.eval = @(x) evaluate(x);
end

% ===== Prox definition =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	if struct % l2-1
		F.prox = @(gam,y) proxL21(gam,y);
	else	  % classic l1
		F.prox = @(gam,y) max((abs(y)-gam*w),0).*sign(y);
	end
end
% -- If the given operator is the gradient operator (1D) -> we use the Chambolle Pock algorithm
if strcmp(A.name,'Operator Gradient') && (A.dim==1)
	% -- Gradient Operator 1D (see below the nested function proxOpGrad)
	F.prox = @(gam,y) proxOpGrad(gam,y);
end

% ===== Prox Fenchel definition =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	% TO DO : Integrer le fait d'avoir des sous indices (variable ind) si c'est possible ...
	F.proxFench = @(gam,y)  (abs(y)<=1).*y + (abs(y)>1).*(sign(y));
end

% ===== Name =====
F.name='Functional Norm 1';

% ===== Nested functions =====
% -- evaluation function
function y=evaluate(x)
	if isempty(ind)
		t=A.eval(x);
	else
		t=A.eval(x(ind.i,ind.j,ind.k));
	end
	y=norm(w(:).*t(:),1);
end 
function y=evaluateStruct(x)
	if isempty(ind)
		t=A.eval(x);
	else
		t=A.eval(x(ind.i,ind.j,ind.k));
	end
	normRows=sqrt(sum(abs(t).^2,2));
	y=norm(w(:).*normRows(:),1);
end 
% -- proximity operator when the gradient operator (1D) is used
function x=proxOpGrad(gam,y)
	if isempty(ind)
		G=FuncLeastSquares([],y);	% Least squares functional
	else
		G=FuncLeastSquares([],y(ind.i,ind.j,ind.k));	% Least squares functional
	end
	F=FuncNorm1([],[]);           % Norm 1 functional
	nK=(A.norm)^2;             % Norm of the gradient operator
	params.tau=0.05;            % algorithm parameters
	params.sig=1/(nK*params.tau)*0.99;
	params.verbose=0;
	params.maxiter=10000;
	params.xTol=1e-4;
	params.FTol=1e-4;
	params.clearTmp=0;
	if isempty(ind)
		[x,~]=AlgoChambollePock(y,F,A,1/gam,G,params);
	else
		x=y;
		[xtemp,~]=AlgoChambollePock(y(ind.i,ind.j,ind.k),F,A,1/gam,G,params);
		x(ind.i,ind.j,ind.k)=xtemp;
	end
end
% -- proximity operator when the l2-1 version is used
function x=proxL21(gam,y)
	normRows=sqrt(sum(abs(y).^2,2));
	x=repmat(max(normRows-gam*w(:),0)./normRows,1,size(y,2)).*y;
end
end
