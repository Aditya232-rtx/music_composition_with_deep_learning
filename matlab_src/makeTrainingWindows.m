function [Xcell, Ycat] = makeTrainingWindows(tokenSeqs, windowLen, numClasses)
%MAKETRAININGWINDOWS Build sliding-window (X,Y) pairs for next-token prediction.
%   tokenSeqs  - cell array of integer token-index column vectors
%   windowLen  - number of previous tokens used as context
%   numClasses - vocabulary size (for fixing the categorical's label set)
%   Xcell      - 1xN cell array, each cell a 1-by-windowLen row (features x time)
%   Ycat       - Nx1 categorical vector of next-token labels
%
%   Preallocated: total window count is computed up front and both output
%   arrays are filled by index rather than grown one element at a time via
%   {end+1} - avoids O(n^2) reallocation once N reaches the hundreds of
%   thousands.

seqLens = cellfun(@numel, tokenSeqs);
windowsPerSeq = max(seqLens - windowLen, 0);
totalWindows = sum(windowsPerSeq);

if totalWindows == 0
    error('makeTrainingWindows:noData', ...
        'No sequences longer than windowLen=%d. Reduce windowLen or add more/longer MIDI files.', windowLen);
end

Xcell = cell(1, totalWindows);
Ylist = zeros(totalWindows, 1);

writeIdx = 0;
for i = 1:numel(tokenSeqs)
    seq = tokenSeqs{i};
    n = windowsPerSeq(i);
    if n == 0
        continue;
    end
    for j = 1:n
        writeIdx = writeIdx + 1;
        Xcell{writeIdx} = seq(j:j+windowLen-1)';
        Ylist(writeIdx) = seq(j+windowLen);
    end
end

Ycat = categorical(Ylist, 1:numClasses);
fprintf('Built %d training windows (windowLen=%d)\n', totalWindows, windowLen);
end
