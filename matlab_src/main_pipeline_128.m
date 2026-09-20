%% Music Composition with Deep Learning - 20-composer / ~128-class experiment
% Variant of main_pipeline.m: fewer composers (20 instead of 58), fewer
% register bands (2 instead of 3) -> ~121-class vocabulary instead of 181.
% Kept as a separate script (rather than overwriting main_pipeline.m) so
% this can be queued to run right after the 181-class/58-composer run
% finishes, without touching that run's files while it's still training.

totalTimer = tic;

%% Config
datasetDir    = 'dataset_subset_20composers';
windowLen     = 32;
outMidiFile   = 'generated_song_20ep_20comp_128class.mid';
genLength     = 200;
temperature   = 1.0;
maxEpochs     = 20;
miniBatchSize = 128;

%% 1. Tokenize the dataset
files = dir(fullfile(datasetDir, '*.mid*'));
assert(~isempty(files), 'No .mid files found in %s - build the subset first.', datasetDir);
midiFiles = fullfile({files.folder}, {files.name});

tTokenize = tic;
[tokenSeqs, vocabMap, bucketEdges, numClasses] = buildTokenDataset128(midiFiles);
fprintf('Tokenization took %.1f s\n', toc(tTokenize));

%% 2. Build supervised training windows
tWindows = tic;
[Xcell, Ycat] = makeTrainingWindows(tokenSeqs, windowLen, numClasses);
fprintf('Window building took %.1f s\n', toc(tWindows));

%% 3. Train the LSTM (supervised next-token classification)
tTrain = tic;
[net, info] = trainMusicLSTM(Xcell', Ycat, numClasses, maxEpochs, miniBatchSize);
fprintf('Training took %.1f s\n', toc(tTrain));
save('trained_music_lstm_20ep_20comp_128class.mat', 'net', 'vocabMap', 'bucketEdges', 'windowLen');
fprintf('Model trained and saved to trained_music_lstm_20ep_20comp_128class.mat\n');

%% 3b. Save training curves + raw history
plotTrainingHistory(info, 'training_curves_20ep_20comp_128class.png');
save('training_info_20ep_20comp_128class.mat', 'info');

%% 4. Generate a new sequence, seeded from a real opening phrase
seed = tokenSeqs{1}(1:min(windowLen, numel(tokenSeqs{1})));
tokenSeq = generateSequence(net, seed, genLength, windowLen, temperature);

%% 5. Decode back to notes and export MIDI
tokensToMidi128(tokenSeq, vocabMap, bucketEdges, outMidiFile);
fprintf('Done. Generated song written to %s\n', outMidiFile);

fprintf('Total pipeline time: %.1f s (%.1f min)\n', toc(totalTimer), toc(totalTimer)/60);
