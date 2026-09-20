function [net, info] = trainMusicLSTM(Xcell, Ycat, numClasses, maxEpochs, miniBatchSize)
%TRAINMUSICLSTM Train a next-token LSTM classifier (supervised, not GAN).
%   Xcell         - Nx1 cell array, each cell a 1-by-windowLen sequence
%   Ycat          - Nx1 categorical vector of next-token labels
%   numClasses    - vocabulary size
%   maxEpochs     - optional, default 18
%   miniBatchSize - optional, default 64 (larger batches reduce iteration
%                   count/overhead on big datasets - bump this up when
%                   training on hundreds of thousands of windows)
%
%   Single-layer, fixed-learning-rate architecture - this was the
%   best-performing config measured so far (beat a stacked+dropout+LR-decay
%   variant at a comparable epoch budget). Kept deliberately unchanged here
%   so the vocabulary-reduction experiment (see buildTokenDataset.m) is an
%   isolated, single-variable comparison against the earlier 20-epoch run.

if nargin < 4, maxEpochs = 18; end
if nargin < 5, miniBatchSize = 64; end

checkpointDir = 'checkpoints';
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end

n = numel(Xcell);
idx = randperm(n);
nVal = max(1, round(0.1*n));
valIdx = idx(1:nVal);
trainIdx = idx(nVal+1:end);

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xcell(valIdx), Ycat(valIdx)}, ...
    'ValidationFrequency', 50, ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'none', ...
    'Verbose', true, ...
    'CheckpointPath', checkpointDir, ...
    'CheckpointFrequency', 1, ...
    'CheckpointFrequencyUnit', 'epoch');
% Plots is 'none' because batch/headless runs can't show a live plot -
% see plotTrainingHistory.m for a saved PNG instead. CheckpointPath saves
% a .mat after every epoch so a killed/interrupted run can resume instead
% of losing all progress.

resumeFile = findLatestCheckpoint(checkpointDir);
if ~isempty(resumeFile)
    fprintf('Resuming from checkpoint: %s\n', resumeFile);
    loaded = load(resumeFile, 'net');
    [net, info] = trainNetwork(Xcell(trainIdx), Ycat(trainIdx), loaded.net.Layers, options);
else
    layers = [
        sequenceInputLayer(1)
        lstmLayer(256, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    [net, info] = trainNetwork(Xcell(trainIdx), Ycat(trainIdx), layers, options);
end
end

function resumeFile = findLatestCheckpoint(checkpointDir)
resumeFile = '';
files = dir(fullfile(checkpointDir, 'net_checkpoint__*.mat'));
if isempty(files)
    return;
end
[~, latestIdx] = max([files.datenum]);
resumeFile = fullfile(files(latestIdx).folder, files(latestIdx).name);
end
