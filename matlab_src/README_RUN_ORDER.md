# Run order

1. **Get data:** download the MAESTRO v3.0.0 **MIDI-only** zip
   (magenta.tensorflow.org/datasets/maestro#v300) and unzip it somewhere local.
2. **Subset it:** in MATLAB —
   ```matlab
   selectMaestroSubset('<unzipped maestro root>', ...
                        '<unzipped maestro root>/maestro-v3.0.0.csv', ...
                        'dataset_subset', 25, 180);
   ```
   This copies ~25 short (<3 min) training-split MIDI files into `dataset_subset/`.
3. **Sanity-check the parser on ONE file first**, before running everything —
   this is the highest-risk part since it's hand-written and untested here:
   ```matlab
   onsets = readMidiNoteOnsets('dataset_subset/<some file>.midi');
   size(onsets)   % should be Nx3, N = number of notes
   onsets(1:10,:) % eyeball: onset times ascending, pitches in 0-127, sane velocities
   ```
   If this errors or returns garbage, fix `readMidiNoteOnsets.m` before going further -
   everything downstream depends on it.
4. **Run the full pipeline:**
   ```matlab
   main_pipeline
   ```
   This tokenizes, trains (~15-20 epochs), generates, and writes `generated_song.mid`.
5. **Check the result:** open `generated_song.mid` in any MIDI player, or drag
   it into GarageBand/MuseScore/similar to listen and eyeball the piano roll.

## Toolbox requirements
- Deep Learning Toolbox (for `trainNetwork`, `lstmLayer`, etc.)
- **No Audio Toolbox needed** — MATLAB has no built-in file-based MIDI reader
  (Audio Toolbox only covers live MIDI device/control-surface I/O), so both
  this project and the prior submission had to hand-roll MIDI parsing.

## Files
| File | Role |
|---|---|
| `readMidiNoteOnsets.m` | Parses a .mid file -> [onset time, pitch, velocity] per note |
| `buildTokenDataset.m` | Tokenizes a folder of MIDI files into (pitch, duration-bucket) vocab indices |
| `makeTrainingWindows.m` | Builds sliding-window (X,Y) pairs for next-token supervision |
| `trainMusicLSTM.m` | Defines + trains the LSTM via `trainNetwork` (supervised, not GAN) |
| `generateSequence.m` | Autoregressive sampling with temperature |
| `tokensToMidi.m` | Decodes generated tokens back into notes |
| `writeMidiFile.m` | Writes a note list to a valid Standard MIDI File |
| `selectMaestroSubset.m` | One-time helper to build the small training subset from MAESTRO's CSV |
| `main_pipeline.m` | Runs everything end to end |

## If something breaks (likely spots, in order of risk)
1. **`readMidiNoteOnsets.m`** — MIDI byte parsing is the riskiest hand-written
   part. If `size(onsets,1)` is 0 or errors, check: file has multiple tracks
   (`numTracks` reads sane, e.g. 1-3), tempo meta event handling, running-status
   handling. Add `disp` statements around the main while-loop if needed.
2. **`makeTrainingWindows.m` error "No sequences longer than windowLen"** —
   means your subset pieces got very few notes after parsing (parser issue) or
   `windowLen=32` is too long for very short pieces — lower it (e.g. 16) or
   pick slightly longer pieces in `selectMaestroSubset`.
3. **Training runs but loss doesn't move** — check `numClasses` isn't absurdly
   large (print it after `buildTokenDataset`); a huge vocabulary from too many
   distinct (pitch,bucket) pairs vs. too little data will make learning slow.
   Coarsen `bucketEdges` in `buildTokenDataset.m` (fewer buckets) if so.
4. **Generated MIDI sounds like noise/silence** — try `temperature` between
   0.7-1.0, and confirm the seed (`tokenSeqs{1}(1:windowLen)`) decodes to
   something sane via `tokensToMidi` on just the seed before trusting full
   generation.

## What's deliberately left out (documented as future work, not bugs)
- Chords / polyphony / multi-instrument generation
- Velocity modeling (fixed at 80 for all generated notes)
- GAN/Transformer architectures
- Training on full MAESTRO rather than a small subset
