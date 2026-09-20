%% Music Composition with Deep Learning - main pipeline
% MathWorks Challenge Project #243
%
% BEFORE RUNNING:
%   1. Download MAESTRO v3.0.0 MIDI-only zip from
%      magenta.tensorflow.org/datasets/maestro#v300 and unzip it locally.
%   2. Run selectMaestroSubset(...) once to build a small 'dataset_subset'
%      folder (see call template below, commented out).
%
% Requires: Deep Learning Toolbox. No Audio Toolbox dependency - MIDI
% file I/O is hand-rolled since MATLAB has no built-in file-based MIDI
% reader (Audio Toolbox only covers live MIDI devices/control surfaces).

%% 0. (Run once) Build the training subset from MAESTRO
% selectMaestroSubset('C:/path/to/maestro-v3.0.0', ...
%                      'C:/path/to/maestro-v3.0.0/maestro-v3.0.0.csv', ...
%                      'dataset_subset', 25, 180);

totalTimer = tic;

%% Config
datasetDir    = 'dataset_subset';   % folder of MAESTRO MIDI files
windowLen     = 32;                 % context length for next-token prediction
outMidiFile   = 'generated_song_20ep_reduced_vocab.mid';
genLength     = 200;                % total tokens in the generated piece
temperature   = 1.0;
maxEpochs     = 20;                 % matches the best-performing prior run
                                     % (single-layer, 885-class) for a clean,
                                     % single-variable vocab-size comparison
miniBatchSize = 128;                % larger batches reduce iteration overhead at this scale

%% 1. Tokenize the dataset
files = dir(fullfile(datasetDir, '*.mid*'));
assert(~isempty(files), 'No .mid files found in %s - run selectMaestroSubset first.', datasetDir);
midiFiles = fullfile({files.folder}, {files.name});

tTokenize = tic;
[tokenSeqs, vocabMap, bucketEdges, numClasses] = buildTokenDataset(midiFiles);
fprintf('Tokenization took %.1f s\n', toc(tTokenize));

%% 2. Build supervised training windows
tWindows = tic;
[Xcell, Ycat] = makeTrainingWindows(tokenSeqs, windowLen, numClasses);
fprintf('Window building took %.1f s\n', toc(tWindows));

%% 3. Train the LSTM (supervised next-token classification)
tTrain = tic;
[net, info] = trainMusicLSTM(Xcell', Ycat, numClasses, maxEpochs, miniBatchSize);
fprintf('Training took %.1f s\n', toc(tTrain));
save('trained_music_lstm_20ep_reduced_vocab.mat', 'net', 'vocabMap', 'bucketEdges', 'windowLen');
fprintf('Model trained and saved to trained_music_lstm_20ep_reduced_vocab.mat\n');

%% 3b. Save training curves + raw history
plotTrainingHistory(info, 'training_curves_20ep_reduced_vocab.png');
save('training_info_20ep_reduced_vocab.mat', 'info');

%% 4. Generate a new sequence, seeded from a real opening phrase
seed = tokenSeqs{1}(1:min(windowLen, numel(tokenSeqs{1})));
tokenSeq = generateSequence(net, seed, genLength, windowLen, temperature);

%% 5. Decode back to notes and export MIDI
tokensToMidi(tokenSeq, vocabMap, bucketEdges, outMidiFile);
fprintf('Done. Generated song written to %s\n', outMidiFile);

fprintf('Total pipeline time: %.1f s (%.1f min)\n', toc(totalTimer), toc(totalTimer)/60);
