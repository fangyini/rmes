% Adds directories to MATLAB path
evm = 'AFER/Eulerian Motion Amplification';
matlab_dir = getenv('MATLAB_CODE');
% Paths for the linear method
addpath(fullfile(matlab_dir,evm, 'Linear'));
addpath(fullfile(matlab_dir,evm, 'matlabPyrTools'));
addpath(fullfile(matlab_dir,evm, 'matlabPyrTools', 'MEX'));

% Paths for the phase-based method
addpath(fullfile(matlab_dir,evm, 'PhaseBased'));
addpath(fullfile(matlab_dir,evm, 'pyrToolsExt'));
addpath(fullfile(matlab_dir,evm, 'Filters'));
