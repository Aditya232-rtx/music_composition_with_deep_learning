function [tokenSeqs, vocabMap, bucketEdges, numClasses] = buildTokenDataset(midiFiles)
%BUILDTOKENDATASET Parse MIDI files and tokenize as (pitch, IOI-bucket) pairs.
%   midiFiles   - cellstr of full paths to .mid files
%   tokenSeqs   - cell array, one integer token-index vector per file
%   vocabMap    - containers.Map: 'pitch_bucket' key -> token index (1 = REST)
%   bucketEdges - inter-onset-interval bucket edges in seconds, shared by
%                 both tokenization here and decoding in tokensToMidi.m
%   numClasses  - size of the vocabulary (max token index)
%
%   Vectorized: (pitch, bucket) pairs are packed into a single integer key
%   (pitch*100 + bucket) and vocab indices are resolved via a direct array
%   lookup instead of a containers.Map hit per note - orders of magnitude
%   faster once the dataset spans hundreds of files / hundreds of
%   thousands of notes.

bucketEdges = [0, 0.06, 0.12, 0.20, 0.30, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, Inf];
numBuckets = numel(bucketEdges) - 1;

rawSeqs = cell(numel(midiFiles), 1);
for i = 1:numel(midiFiles)
    try
        onsets = readMidiNoteOnsets(midiFiles{i}); % [time, pitch, vel]
        if size(onsets,1) < 2
            continue;
        end
        t = onsets(:,1);
        pitch = onsets(:,2);
        ioi = [0.06; diff(t)];
        bucketIdx = discretize(ioi, bucketEdges);
        rawSeqs{i} = [pitch, bucketIdx];
        fprintf('Parsed %s: %d notes\n', midiFiles{i}, numel(pitch));
    catch err
        fprintf('Skipping %s (%s)\n', midiFiles{i}, err.message);
    end
end

% ---- Build vocabulary over all (pitch, bucket) pairs actually seen ----
% key space is bounded (pitch 0-127, bucket 1-numBuckets), so a direct
% array lookup replaces per-note containers.Map get/set calls.
maxKey = 127*100 + numBuckets;
keyToIdx = zeros(maxKey, 1);
nextIdx = 2; % 1 is reserved for REST

for i = 1:numel(rawSeqs)
    seq = rawSeqs{i};
    if isempty(seq), continue; end
    keys = seq(:,1)*100 + seq(:,2);
    uniqueKeys = unique(keys);
    newKeys = uniqueKeys(keyToIdx(uniqueKeys) == 0);
    numNew = numel(newKeys);
    if numNew > 0
        keyToIdx(newKeys) = nextIdx:(nextIdx+numNew-1);
        nextIdx = nextIdx + numNew;
    end
end
numClasses = nextIdx - 1;

% ---- Build vocabMap ('pitch_bucket' -> index) for tokensToMidi decoding ----
vocabMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
vocabMap('REST') = 1;
seenKeys = find(keyToIdx > 0);
for k = 1:numel(seenKeys)
    key = seenKeys(k);
    pitchVal = floor(key/100);
    bucketVal = key - pitchVal*100;
    vocabMap(sprintf('%d_%d', pitchVal, bucketVal)) = keyToIdx(key);
end

% ---- Convert each file's sequence to token indices (vectorized) ----
tokenSeqs = cell(numel(rawSeqs), 1);
for i = 1:numel(rawSeqs)
    seq = rawSeqs{i};
    if isempty(seq), continue; end
    keys = seq(:,1)*100 + seq(:,2);
    tokenSeqs{i} = keyToIdx(keys);
end
tokenSeqs = tokenSeqs(~cellfun(@isempty, tokenSeqs));

fprintf('Vocabulary size: %d | usable sequences: %d\n', numClasses, numel(tokenSeqs));
end
