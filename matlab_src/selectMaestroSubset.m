function selectMaestroSubset(maestroRoot, csvPath, outDir, numPieces, maxDurationSec)
%SELECTMAESTROSUBSET Copy a small subset of short MAESTRO training pieces.
%   Run this once AFTER manually downloading + unzipping the MAESTRO
%   v3.0.0 MIDI-only archive (magenta.tensorflow.org/datasets/maestro#v300).
%
%   maestroRoot    - folder where the unzipped MAESTRO contents live
%                    (contains maestro-v3.0.0.csv and year subfolders)
%   csvPath        - full path to maestro-v3.0.0.csv
%   outDir         - folder to copy the chosen subset's .mid files into
%   numPieces      - how many pieces to keep (default 25)
%   maxDurationSec - only consider pieces shorter than this (default 180)

if nargin < 4, numPieces = 25; end
if nargin < 5, maxDurationSec = 180; end

T = readtable(csvPath);
% maestro-v3.0.0.csv columns include: canonical_composer, canonical_title,
% split, year, midi_filename, audio_filename, duration

isTrain = strcmp(T.split, 'train');
isShort = T.duration <= maxDurationSec;
candidates = T(isTrain & isShort, :);
candidates = sortrows(candidates, 'duration');

if height(candidates) < numPieces
    warning('Only %d pieces match the filter; using all of them.', height(candidates));
    numPieces = height(candidates);
end
chosen = candidates(1:numPieces, :);

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

for i = 1:height(chosen)
    src = fullfile(maestroRoot, chosen.midi_filename{i});
    [~, name, ext] = fileparts(src);
    dst = fullfile(outDir, [name ext]);
    copyfile(src, dst);
end

fprintf('Copied %d MIDI files (<= %ds each) to %s\n', height(chosen), maxDurationSec, outDir);
end
