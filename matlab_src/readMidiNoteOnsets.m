function onsets = readMidiNoteOnsets(filePath)
%READMIDINOTEONSETS Parse a Standard MIDI File and return NoteOn events.
%   onsets = readMidiNoteOnsets(filePath) returns an Nx3 matrix:
%       [onsetTimeSeconds, pitch, velocity]
%   sorted by onset time, merged across all tracks. NoteOn events with
%   velocity 0 (the running-status convention for NoteOff) are excluded.
%   Only tempo (0xFF 0x51) meta events are used; all other meta/sysex
%   events are parsed just enough to skip over correctly.

fid = fopen(filePath, 'r');
if fid == -1
    error('readMidiNoteOnsets:openFailed', 'Cannot open file: %s', filePath);
end
raw = fread(fid, Inf, 'uint8=>uint8');
fclose(fid);

pos = 1; % 1-based pointer into raw

% ---- Header chunk ----
if numel(raw) < 14 || ~strcmp(char(raw(pos:pos+3)'), 'MThd')
    error('readMidiNoteOnsets:badHeader', 'Not a valid MIDI file (missing MThd): %s', filePath);
end
pos = pos + 4;
pos = pos + 4; % header length (always 6), skip
pos = pos + 2; % format type, skip
numTracks  = readUint16(raw, pos); pos = pos + 2;
division   = readUint16(raw, pos); pos = pos + 2;

if division >= 32768
    error('readMidiNoteOnsets:smpte', 'SMPTE-based division not supported: %s', filePath);
end
ticksPerQuarter = double(division);

allEvents = struct('tick', {}, 'type', {}, 'pitch', {}, 'velocity', {}, 'usPerQuarter', {});

for tr = 1:numTracks
    if ~strcmp(char(raw(pos:pos+3)'), 'MTrk')
        error('readMidiNoteOnsets:badTrack', 'Expected MTrk chunk in %s', filePath);
    end
    pos = pos + 4;
    trackLen = readUint32(raw, pos); pos = pos + 4;
    trackEnd = pos + trackLen - 1;

    tick = 0;
    runningStatus = 0;

    while pos <= trackEnd
        [delta, pos] = readVarLen(raw, pos);
        tick = tick + delta;

        statusByte = raw(pos);
        if statusByte >= 128
            runningStatus = statusByte;
            pos = pos + 1;
        else
            statusByte = runningStatus; % running status; current byte is already data
        end

        highNibble = bitshift(uint16(statusByte), -4);

        if statusByte == 255
            metaType = raw(pos); pos = pos + 1;
            [len, pos] = readVarLen(raw, pos);
            metaData = raw(pos:pos+len-1); pos = pos + len;
            if metaType == 81 && len == 3 % Set Tempo
                usPerQuarter = double(metaData(1))*65536 + double(metaData(2))*256 + double(metaData(3));
                e = struct('tick', tick, 'type', 'tempo', 'pitch', 0, 'velocity', 0, 'usPerQuarter', usPerQuarter);
                allEvents(end+1) = e; %#ok<AGROW>
            end
        elseif statusByte == 240 || statusByte == 247
            [len, pos] = readVarLen(raw, pos);
            pos = pos + len;
        elseif highNibble == 9 || highNibble == 8
            pitch = raw(pos); vel = raw(pos+1); pos = pos + 2;
            if highNibble == 9 && vel > 0
                e = struct('tick', tick, 'type', 'on', 'pitch', double(pitch), 'velocity', double(vel), 'usPerQuarter', 0);
                allEvents(end+1) = e; %#ok<AGROW>
            end
        else
            nData = channelMessageDataBytes(highNibble);
            pos = pos + nData;
        end
    end
    pos = trackEnd + 1;
end

if isempty(allEvents)
    onsets = zeros(0,3);
    return;
end

% ---- Build tempo map (tick -> microseconds/quarter), default 500000 @ tick 0 ----
tempoEvents = allEvents(strcmp({allEvents.type}, 'tempo'));
tempoTicks = [tempoEvents.tick];
tempoValues = [tempoEvents.usPerQuarter];
if isempty(tempoTicks) || tempoTicks(1) > 0
    tempoTicks = [0, tempoTicks];
    tempoValues = [500000, tempoValues];
end
[tempoTicks, order] = sort(tempoTicks);
tempoValues = tempoValues(order);

noteEvents = allEvents(strcmp({allEvents.type}, 'on'));
noteTicks = [noteEvents.tick];
[noteTicks, order] = sort(noteTicks);
noteEvents = noteEvents(order);

if isempty(noteEvents)
    onsets = zeros(0,3);
    return;
end

onsetSeconds = tickToSeconds(noteTicks, tempoTicks, tempoValues, ticksPerQuarter);
onsets = [onsetSeconds(:), [noteEvents.pitch]', [noteEvents.velocity]'];
onsets = sortrows(onsets, 1);
end

% ---------------- local helpers ----------------

function n = channelMessageDataBytes(highNibble)
switch highNibble
    case {8,9,10,11,14}
        n = 2; % note off/on, poly aftertouch, control change, pitch bend
    case {12,13}
        n = 1; % program change, channel aftertouch
    otherwise
        n = 2;
end
end

function v = readUint32(raw, pos)
v = double(raw(pos))*16777216 + double(raw(pos+1))*65536 + double(raw(pos+2))*256 + double(raw(pos+3));
end

function v = readUint16(raw, pos)
v = double(raw(pos))*256 + double(raw(pos+1));
end

function [value, newPos] = readVarLen(raw, pos)
value = 0;
while true
    b = raw(pos);
    pos = pos + 1;
    value = bitshift(value, 7) + double(bitand(b, uint8(127)));
    if bitand(b, uint8(128)) == 0
        break;
    end
end
newPos = pos;
end

function seconds = tickToSeconds(ticks, tempoTicks, tempoValues, ticksPerQuarter)
% Piecewise-linear tick->second conversion honoring tempo changes.
seconds = zeros(size(ticks));
for i = 1:numel(ticks)
    tk = ticks(i);
    segIdx = find(tempoTicks <= tk, 1, 'last');
    accumSec = 0;
    for s = 1:segIdx-1
        segTicks = tempoTicks(s+1) - tempoTicks(s);
        accumSec = accumSec + segTicks * (tempoValues(s) / 1e6) / ticksPerQuarter;
    end
    remTicks = tk - tempoTicks(segIdx);
    accumSec = accumSec + remTicks * (tempoValues(segIdx) / 1e6) / ticksPerQuarter;
    seconds(i) = accumSec;
end
end
