function [Op]=OperatorGradient(dim,varargin)
%------------------------------------------------
% function [Op]=OperatorGradient(dim,varargin)
%
% Usage : Op=OperatorGradient(dim)
%         Op=OperatorGradient(dim,resol)
%
% Constructs the gradient operator operator
%
% Inputs : dim      -> dimension: 1 (1D), 2 (2D) or 3 (3D)
%          varargin -> resol : resolution of the image, structure with fields
%                        .dx (default 1)
%                        .dy (if 2D or 3D, default: 1)
%                        .dz (if 3D, default: 1)
%          
% Output : Op    -> Identity Operator: structure with fields
%                    .eval   -> Op.eval(x) evaluates the operator at x
%                    .adj    -> Op.adj(x) evaluates the adjoint at x 
%                    .resol  -> the used resolutions
%                    .norm   -> norm of the operator ||A|| (upper bound)                             
%                    .name   -> name of the operator
%                    .dim    -> dimension of the variable
%
% Note : In 1D the returned operator works on column vectors
%
% -- Emmanuel Soubies (2015)
%------------------------------------------------

% -- Get the optional parameter
narg=nargin;
if narg==1
	resol.dx=1; resol.dy=1; resol.dz=1;
elseif narg==2
	resol=varargin{1};
elseif nargin > 2 
	error('In OperatorGradient : Wrong number of inputs arguments !');
end

if dim==1
% ===== 1D case =====
	% --- Operator Definition 
	Op.eval = @(x) [(x(2:end)-x(1:end-1));0]/resol.dx;
	% --- Adjoint Definition 
	Op.adj = @(x) [[-x(1) ; (-x(2:end-1)+x(1:end-2))] ; x(end-1)]/resol.dx;
	% --- Norm of the operator (upper bound)
	Op.norm = 2/resol.dx;
elseif dim==2
% ===== 2D case =====
	% --- Operator Definition 
	Op.eval = @(x) cat(3,([x(:,2:end),x(:,end)]-x)/resol.dx,...
						 ([x(2:end,:);x(end,:)]-x)/resol.dy);
	% --- Adjoint Definition 
	Op.adj = @(x) [-x(:,1,1),...
				   -x(:,2:end-1,1)+x(:,1:end-2,1),...
				    x(:,end-1,1)]/resol.dx...
				 +[-x(1,:,2);...
				   -x(2:end-1,:,2)+x(1:end-2,:,2);...
					x(end-1,:,2)]/resol.dy;
	% --- Norm of the operator (upper bound)
	Op.norm = 2*sqrt(1/resol.dx^2+1/resol.dy^2);
elseif dim==3
% ===== 3D case =====
	% --- Operator Definition 
	Op.eval = @(x) cat(4,([x(:,2:end,:),x(:,end,:)]-x)/resol.dx,...
			             ([x(2:end,:,:);x(end,:,:)]-x)/resol.dy,...
			             (cat(3,x(:,:,2:end),x(:,:,end))-x)/resol.dz);
	% --- Adjoint Definition 
	Op.adj = @(x) [-x(:,1,:,1),...
				   -x(:,2:end-1,:,1)+x(:,1:end-2,:,1),...
                    x(:,end-1,:,1)]/resol.dx...
                + [-x(1,:,:,2);...
                   -x(2:end-1,:,:,2)+x(1:end-2,:,:,2);...
                    x(end-1,:,:,2)]/resol.dy...
                + cat(3,-x(:,:,1,3),...
                        -x(:,:,2:end-1,3)+x(:,:,1:end-2,3),...
                         x(:,:,end-1,3))/resol.dz;
	% --- Norm of the operator (upper bound)
	Op.norm = 2*sqrt(1/resol.dx^2+1/resol.dy^2+1/resol.dz^2);
else
	error('In OperatorGradient : parameter dim must be equal to 1,2 or 3');
end

% ===== Name =====
Op.name='Operator Gradient';

% ===== Dimension =====
Op.dim=dim;

% ===== Resolutions =====
Op.resol=resol;
end
