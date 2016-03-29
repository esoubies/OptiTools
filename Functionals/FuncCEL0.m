function [F]=FuncCEL0(nai,lamb,struct)
%------------------------------------------------
% function [F]=FuncCEL0(nai,lamb,struct)
%
% Constructs the CEL0 functional [1]
%
%  P(x) = lamb - 0.5*a^2*(|x| - sqrt(2*lamb)/a)^2*1_{|x| <= sqrt(2*lamb)/a}  (in 1D : ND => sum of 1D)
%
% where 'a' stands for the column of the operator of the quadratic term associated to the 
% considered variable and lamb >0.
%								
% Inputs : nai    -> norm of the columns of the operator of the associated quadratic data term
%                    same size as the variable. If struct is activated then same size as the 
%                    number of rows of the variable
%          lamb   -> parameter lambda (real)
%          struct -> boolean: if true (1) then the row-sparsity CEL0 is used,  that is when the variable is a 
%                    matrix and we seek for a solution sparse w.r.t the rows. Otherwise the standard CEL0
%                    penalty is used. (default 0)
%
%          Note : when the variable is a vector the row-structured or the standard version are the same.
%
% Output : F -> Structured raw sparsity CEL0 penalty: structure with fields
%                 .eval      -> F.eval(x) evaluates the functional at x
%                 .prox      -> F.prox(gam,x) evaluates the proximity operator at x  
%                 .subgrad   -> F.subgrad(x) return a subgradient at x                                   
%                 .name      -> name of the functional
%
% References :
%    [1] E. Soubies, L. Blanc-Feraud, G. Aubert. A Continuous Exact l0 penalty (CEL0) for Least Squares Regularized
%        Problem SIAM Journal on Imaging Science. 8(3):1607-1639, 2015.
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------
if isempty(struct)
	struct=0;   % default value
end;

% -- Precomputed variables
bound=sqrt(2*lamb)./nai;
bound2=sqrt(2*lamb).*nai;
nai2=nai.^2;
coef=0.5*nai.^2;

% ===== Function definition =====
if struct % row-sparsity CEL0
	F.eval = @(X) evaluateStructCEL0(X);
else      % classical CEL0
	F.eval = @(x) numel(x)*lamb - sum(coef.*(abs(x) - bound).^2.*(abs(x) <= bound));
end

% ===== Prox definition =====
if struct % row-sparsity CEL0
	F.prox = @(gam,y) proxStructCEL0(gam,y);
else      % classical CEL0
	F.prox = @(gam,y) proxCEL0(gam,y);
end

% ===== Subgrad definition =====
if ~struct  % classical CEL0
	F.subgrad = @(x) (abs(x)>=eps).*(abs(x)<bound).*(-nai2.*x+sign(x).*bound2) + (abs(x)<eps).*bound2;
end

% ===== Name =====
if struct  % row-sparsity CEL0
	F.name='Functional Structured row CEL0';
else       % classical CEL0
	F.name='Functional CEL0';
end

% ===== Nested functions =====
% -- Evaluate
function y=evaluateStructCEL0(X)
	normRows=sqrt(sum(abs(X).^2,2));
	y=size(X,1)*lamb - sum(coef.*(normRows - bound).^2.*(normRows <= bound));
end
% -- Proximity operator
function x=proxCEL0(gam,y)
    absy=abs(y);
    int=nai.^2*gam <1; 
    num=(absy - bound2*gam); 
    num(num<0)=0; 
    x=int.*sign(y).*min(absy, num./(1-nai2*gam))  + (1-int).*y.*(absy > sqrt(2*gam*lamb));
    % note : when y is complex sign(y)=y/abs(y) if y \neq 0 and 0 otherwise
end
function x=proxStructCEL0(gam,y)
	int=repmat((nai.^2*gam <1),1,size(y,2)); 
	normRows=repmat(sqrt(sum(abs(y).^2,2)),1,size(y,2));
	signy=y./normRows; signy(normRows==0)=0;
	naiMat=repmat(nai,1,size(y,2)); 
	num=(normRows - sqrt(2*lamb)*naiMat*gam); 
	num(num<0)=0; 
    x=int.*signy.*min(normRows, num./(1-naiMat.^2*gam))  + (1-int).*y.*(normRows > sqrt(2*gam*lamb));
end
end
