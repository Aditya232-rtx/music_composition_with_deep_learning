function tokensToMidi128(tokenSeq, vocabMap, bucketEdges, outFile)
%TOKENSTOMIDI128 Decode a generated token sequence into notes and write a MIDI file.
%   Paired with buildTokenDataset128.m's 2-register-band scheme.

% Build reverse lookup: token index -> 'pitchClass_band_bucket' (or 'REST')
keysList = keys(vocabMap);
valsList = values(vocabMap);
reverseMap = containers.Map('KeyType','double','ValueType','char');
for i = 1:numel(keysList)
    reverseMap(valsList{i}) = keysList{i};
end

% Representative duration (seconds) for each bucket; last (Inf-ended)
% bucket is capped to edges(end-1)+1.0 rather than Inf.
bucketSeconds = 0.5*(bucketEdges(1:end-1) + min(bucketEdges(2:end), bucketEdges(end-1)+1.0));

notes = struct('pitch', {}, 'onset', {}, 'duration', {}, 'velocity', {});
t = 0;
for i = 1:numel(tokenSeq)
    if ~isKey(reverseMap, tokenSeq(i))
        continue;
    end
    key = reverseMap(tokenSeq(i));
    if strcmp(key, 'REST')
        t = t + 0.3;
        continue;
    end
    parts = sscanf(key, '%d_%d_%d');
    pitchClass = parts(1);
    band = parts(2);
    bucket = parts(3);
    pitch = bandToAnchorPitch(band) + pitchClass;
    ioi = bucketSeconds(bucket);
    if i > 1
        t = t + ioi;
    end
    notes(end+1) = struct('pitch', pitch, 'onset', t, ...
        'duration', max(0.08, ioi*0.85), 'velocity', 80); %#ok<AGROW>
end

if isempty(notes)
    error('tokensToMidi128:noNotes', 'Decoded zero notes - check vocabMap/tokenSeq consistency.');
end

writeMidiFile(notes, outFile);
end

function pitch = bandToAnchorPitch(band)
%BANDTOANCHORPITCH Representative base MIDI pitch for a register band
%   (matches the low/high split in buildTokenDataset128.m's pitchToBand).
anchors = [48, 72]; % band 0=low, 1=high
pitch = anchors(band + 1);
end
