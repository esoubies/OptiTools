function [F]=FuncLeastSquares(A,d)
%------------------------------------------------
% function [F]=FuncLeastSquares(A,d)
%
% Constructs the functional: 0.5*||Ax-d||^2 where A is a given operator
% and d represents the data
%
% Inputs : A -> Operator (see the Operators folder, default : Identity)
%          d -> data
%
% Output : F -> Least Squares functional: structure with fields
%                 .eval      -> F.eval(x) evaluates the functional at x
%                 .lip       -> Lipschitz constant of the gradient (if implemented for the given operator)
%                 .lipFench  -> Lipschitz constant of the gradient of the Fenchel transform (if 
%                               implemented for the given operator)
%                 .grad      -> F.grad(x) evaluates the gradient at x (if the 
%                               gradient is implemented for the given operator)   
%                 .prox      -> F.prox(gam,x) evaluates the proximity operator at x (if it 
%                               is implemented for the given operator)      
%                 .proxFench -> F.proxFench(gam,x) evaluates the proximity operator of the
%                               Fenchel Transform at x (if it is implemented for the given operator)                                  
%                 .name      -> name of the functional
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------
global R gAd
% -- If no operator is given, the Identity is setted
if isempty(A)
	A=OperatorIdentity();
end

% -- Precomputed variables
if isfield(A,'adj')
	Ad=A.adj(d);
end

% ===== Function definition =====
F.eval = @(x) 0.5*norm(reshape(A.eval(x)-d,[],1),'fro')^2;

% ===== Gradient definition =====
% -- If the given operator is linear
if isfield(A,'adj')
	if isfield(A,'AtA')
		F.grad = @(x) A.AtA(x)-Ad;	
	else
		F.grad = @(x) A.adj(A.eval(x)-d);
	end
end

% ===== Prox definition =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	F.prox = @(gam,y) (y + gam*d)/(gam+1);
end
% -- If the given operator is the TIRF-1D operator
if strcmp(A.name,'Operator TIRF-1D')
	F.prox = @(gam,y) proxOpTIRF1D(gam,y);
end
% -- If the given operator is the Convolution operator
if strcmp(A.name,'Operator Convolution') 
	if isvector(A.kernel)
	else
		H=fft2(A.kernel);        % Fourier Transform of the kernel K
		H2=abs(H).^2;
		fftdHe=conj(H).*fft2(d);
		F.prox = @(gam,y)  real(ifft2((gam*fftdHe  + fft2(y))./(gam*H2+1)));
	end
end

% ===== Prox Fenchel definition =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	F.proxFench = @(gam,y) (y-gam*d)/(gam+1);
end

% ===== Lipschitz constant of the gradient =====
if isfield(A,'norm')
    F.lip=A.norm^2;
end

% ===== Lipschitz constant of the gradient of the Fenchel transform =====
% -- If the given operator is the identity operator
if strcmp(A.name,'Operator Identity')
	F.lipFench=1;
end

% ===== Name =====
F.name=['Functional Least Squares combined with ',A.name];

% ===== Nested functions =====
% -- proximity operator when the TIRF-1D operator is used
if strcmp(A.name,'Operator TIRF-1D')
	gamOld=0;
	M=A.mat;
	AA=M'*M;
end
function x=proxOpTIRF1D(gam,y)
	if gam~=gamOld	
		gAd=gam*M'*d;
		R = inv(eye(size(AA)) + gam*AA);
		gamOld=gam;
	end
	x=R*(y+gAd);
end
end
