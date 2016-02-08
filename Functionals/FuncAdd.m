function [F]=FuncAdd(F1,F2,w)
%------------------------------------------------
% function [F]=FuncAdd(F1,F2,w)
%
% Constructs a functional which is the sum of the two given functionals.
%
% Inputs : F1  -> functional 1
%          F2  -> functional 2
%          w   -> weight of the second functional (F1 + w*F2)
%
% Output : F -> Mixed (2,1) Regularized Norm: structure with fields
%                 .eval   -> F.eval(x) evaluates F1(x) + F2(x)
%                 .grad   -> F.grad(x) evaluates Grad(F1(x)) + Grad(F2(x))
%                            (if the two functionals are differentiables)   
%                 .lip       -> Lipschitz constant of the gradient (if the two functionals 
%                               are gradient-Lipschitz)                          
%                 .name   -> name of the functional
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% ===== Function definition =====
F.eval = @(x) F1.eval(x) + w*F2.eval(x);

% ===== Gradient definition =====
% if the two fonctionals are differentiables
if isfield(F1,'grad') && isfield(F2,'grad')
	F.grad = @(x) F1.grad(x) + w*F2.grad(x);
end

% ===== Lipschitz constant of the gradient =====
if isfield(F1,'lip') && isfield(F2,'lip')
	F.lip=F1.lip + w*F2.lip;
end

% ===== Name =====
F.name=['Add functionals : ',F1.name,' and ',F2.name, ' (with weight ',num2str(w),')'];
