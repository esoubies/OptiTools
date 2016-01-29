%----------------------------------------------
% This installation script updates (and saves) the path
% of Matlab in order to be able to access to the
% OptiTools functions.
%
% Note : you need to be in root mode to run this script.
%
% Emmanuel Soubies (2016)
%----------------------------------------------

% -- Get the current folder path
dir=cd;
% -- Update the Matlab path
addpath([dir,'/Algorithms']);
addpath([dir,'/Functional']);
addpath([dir,'/Operators']);
addpath([dir,'/Examples']);

% -- Save the updated path
savepath;
