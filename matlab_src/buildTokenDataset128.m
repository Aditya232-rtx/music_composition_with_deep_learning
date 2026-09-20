function [tokenSeqs, vocabMap, bucketEdges, numClasses] = buildTokenDataset128(midiFiles)
%BUILDTOKENDATASET128 Parse MIDI files and tokenize as (pitch-class, register-band, IOI-bucket) triples.
%   Same scheme as buildTokenDataset.m but with 2 register bands instead
%   of 3 (12 pitch classes x 2 bands x 5 duration buckets + 1 REST = 121
%   classes) - a further vocabulary reduction experiment, paired with a
%   20-composer dataset subset (dataset_subset_20composers/) instead of
%   the full 58-composer one.

bucketEdges = [0, 0.15, 0.35, 0.7, 1.5, Inf];
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
        pitchClass = mod(pitch, 12);
        band = pitchToBand(pitch);
        ioi = [0.06; diff(t)];
        bucketIdx = discretize(ioi, bucketEdges);
        rawSeqs{i} = [pitchClass, band, bucketIdx];
        fprintf('Parsed %s: %d notes\n', midiFiles{i}, numel(pitch));
    catch err
        fprintf('Skipping %s (%s)\n', midiFiles{i}, err.message);
    end
end

maxKey = 11*1000 + 1*100 + numBuckets;
keyToIdx = zeros(maxKey, 1);
nextIdx = 2; % 1 is reserved for REST

for i = 1:numel(rawSeqs)
    seq = rawSeqs{i};
    if isempty(seq), continue; end
    keys = seq(:,1)*1000 + seq(:,2)*100 + seq(:,3);
    uniqueKeys = unique(keys);
    newKeys = uniqueKeys(keyToIdx(uniqueKeys) == 0);
    numNew = numel(newKeys);
    if numNew > 0
        keyToIdx(newKeys) = nextIdx:(nextIdx+numNew-1);
        nextIdx = nextIdx + numNew;
    end
end
numClasses = nextIdx - 1;

vocabMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
vocabMap('REST') = 1;
seenKeys = find(keyToIdx > 0);
for k = 1:numel(seenKeys)
    key = seenKeys(k);
    pitchClassVal = floor(key/1000);
    rem1 = key - pitchClassVal*1000;
    bandVal = floor(rem1/100);
    bucketVal = rem1 - bandVal*100;
    vocabMap(sprintf('%d_%d_%d', pitchClassVal, bandVal, bucketVal)) = keyToIdx(key);
end

tokenSeqs = cell(numel(rawSeqs), 1);
for i = 1:numel(rawSeqs)
    seq = rawSeqs{i};
    if isempty(seq), continue; end
    keys = seq(:,1)*1000 + seq(:,2)*100 + seq(:,3);
    tokenSeqs{i} = keyToIdx(keys);
end
tokenSeqs = tokenSeqs(~cellfun(@isempty, tokenSeqs));

fprintf('Vocabulary size: %d | usable sequences: %d\n', numClasses, numel(tokenSeqs));
end

function band = pitchToBand(pitch)
%PITCHTOBAND Collapse absolute MIDI pitch into a coarse register: 0=low, 1=high.
band = zeros(size(pitch));
band(pitch >= 60) = 1;
end
