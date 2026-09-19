function writeMidiFile(notes, outFile, ticksPerQuarter)
%WRITEMIDIFILE Write a struct array of notes to a Type-0 Standard MIDI File.
%   notes - struct array with fields: pitch, onset (sec), duration (sec), velocity
%   outFile - output .mid path
%   ticksPerQuarter - optional, default 480

if nargin < 3, ticksPerQuarter = 480; end
if isempty(notes)
    error('writeMidiFile:empty', 'No notes to write.');
end

tempoUsPerQuarter = 500000; % fixed 120 BPM
secToTick = @(s) round(s * (1e6/tempoUsPerQuarter) * ticksPerQuarter);

events = zeros(numel(notes)*2, 4); % [tick, type(1=on,0=off), pitch, velocity]
r = 1;
for i = 1:numel(notes)
    n = notes(i);
    onTick = secToTick(n.onset);
    offTick = secToTick(n.onset + n.duration);
    if offTick <= onTick
        offTick = onTick + 1; % guarantee non-zero duration
    end
    events(r,:)   = [onTick, 1, n.pitch, n.velocity]; r = r + 1;
    events(r,:)   = [offTick, 0, n.pitch, 0];         r = r + 1;
end
% Sort by tick; break ties with NoteOff (0) before NoteOn (1) so a note
% ending and the next one starting at the same tick don't get stuck.
events = sortrows(events, [1, 2]);

trackBytes = uint8([]);
prevTick = 0;
trackBytes = [trackBytes, writeVarLen(0), ...
    uint8([255, 81, 3, bitshift(tempoUsPerQuarter,-16), ...
           bitand(bitshift(tempoUsPerQuarter,-8),255), bitand(tempoUsPerQuarter,255)])];

for i = 1:size(events,1)
    deltaTick = events(i,1) - prevTick;
    prevTick = events(i,1);
    trackBytes = [trackBytes, writeVarLen(deltaTick)]; %#ok<AGROW>
    if events(i,2) == 1
        statusByte = uint8(144); % NoteOn, channel 1
    else
        statusByte = uint8(128); % NoteOff, channel 1
    end
    trackBytes = [trackBytes, statusByte, uint8(events(i,3)), uint8(events(i,4))]; %#ok<AGROW>
end
trackBytes = [trackBytes, writeVarLen(0), uint8([255,47,0])]; % End of Track

fid = fopen(outFile, 'w', 'ieee-be');
if fid == -1
    error('writeMidiFile:openFailed', 'Cannot open %s for writing', outFile);
end
fwrite(fid, 'MThd', 'char');
fwrite(fid, 6, 'uint32');
fwrite(fid, 1, 'uint16');
fwrite(fid, 1, 'uint16');
fwrite(fid, ticksPerQuarter, 'uint16');
fwrite(fid, 'MTrk', 'char');
fwrite(fid, numel(trackBytes), 'uint32');
fwrite(fid, trackBytes, 'uint8');
fclose(fid);
fprintf('MIDI file written to: %s\n', outFile);
end

function bytes = writeVarLen(value)
if value < 0, value = 0; end
buf = uint8(bitand(value, 127));
value = bitshift(value, -7);
while value > 0
    buf = [uint8(bitor(bitand(value,127), 128)), buf]; %#ok<AGROW>
    value = bitshift(value, -7);
end
bytes = buf;
end
