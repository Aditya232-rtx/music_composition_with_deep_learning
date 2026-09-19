function tokensToMidi(tokenSeq, vocabMap, bucketEdges, outFile)
%TOKENSTOMIDI Decode a generated token sequence into notes and write a MIDI file.
%   tokenSeq    - vector of token indices (from generateSequence)
%   vocabMap    - the containers.Map produced by buildTokenDataset
%   bucketEdges - the same bucket edges used to build vocabMap
%   outFile     - output .mid path

% Build reverse lookup: token index -> 'pitch_bucket' (or 'REST')
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
    parts = sscanf(key, '%d_%d');
    pitch = parts(1);
    bucket = parts(2);
    ioi = bucketSeconds(bucket);
    if i > 1
        t = t + ioi;
    end
    notes(end+1) = struct('pitch', pitch, 'onset', t, ...
        'duration', max(0.08, ioi*0.85), 'velocity', 80); %#ok<AGROW>
end

if isempty(notes)
    error('tokensToMidi:noNotes', 'Decoded zero notes - check vocabMap/tokenSeq consistency.');
end

writeMidiFile(notes, outFile);
end
