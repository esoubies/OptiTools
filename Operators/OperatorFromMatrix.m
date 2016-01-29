function [Op]=OperatorFromMatrix(A)
%------------------------------------------------
% function [Op]=OperatorFromMatrix(A)
%
% Constructs an operator defined by the given matrix 
%
% Input :  A  -> matrix defining the operator
%
% Output : Op -> Identity Operator: structure with fields
%                 .eval   -> Op.eval(x) evaluates the operator at x
%                 .adj    -> Op.adj(x) evaluates the adjoint at x     
%                 .AtA    -> Op.AtA(x) evaluates the adjoint at Op.eval(x)     
%                 .nAtA   -> norm of A^tA 
%                 .norm   -> norm of the operator ||A||                        
%                 .name   -> name of the operator
%                 .nai    -> raw vector containing the norm of the columns of A
%
%          Note : works also for complex valued A
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% Note : when A is complex valued, A' provides the conjugate-transpose (one
% has to use A.' to have only the transpose so: A'=conj(A).')

% -- Precomputed variables
AA=A'*A;

% ===== Operator Definition =====
Op.eval=@(x) A*x;

% ===== Adjoint Definition =====
Op.adj=@(x) A'*x;

% ===== AtA Definition =====
Op.AtA=@(x) AA*x;

% ===== Norm of AtA =====
Op.nAtA=norm(A'*A);

% ===== Norm of A =====
Op.norm=norm(A);

% ===== Norm of the columns of A =====
Op.nai=sqrt(sum(abs(A).^2,1));

% ===== Name =====
Op.name='Operator From Matrix';
end
