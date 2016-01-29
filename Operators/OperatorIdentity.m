function [Op]=OperatorIdentity()
%------------------------------------------------
% function [Op]=OperatorIdentity()
%
% Constructs the identity operator
%
% Output : Op -> Identity Operator: structure with fields
%                 .eval   -> Op.eval(x) evaluates the operator at x
%                 .adj    -> Op.adj(x) evaluates the adjoint at x  
%                 .AtA    -> Op.AtA(x) evaluates the adjoint at Op.eval(x)     
%                 .norm   -> norm of the operator ||A||                              
%                 .name   -> name of the operator
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% ===== Operator Definition =====
Op.eval=@(x) x;

% ===== Adjoint Definition =====
Op.adj=@(x) x;

% ===== AtA Definition =====
Op.AtA=@(x) x;

% ===== Norm of A =====
Op.norm=1;

% ===== Name =====
Op.name='Operator Identity';

end
