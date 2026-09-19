# Music Composition with Deep Learning — MATLAB Build Spec
MathWorks Challenge Project #243 | Deadline: tonight, midnight IST

## Goal
Supervised LSTM that learns to predict the next note in a sequence, trained on a
MAESTRO subset, generating a new playable MIDI file. Beat the one existing
submission (a noise-conditioned LSTM-GAN that ignores duration/velocity at
generation time) by actually using duration end-to-end and training with real
supervision instead of adversarial noise.

## Dataset
- Source: MAESTRO v3.0.0, **MIDI-only zip** (no audio) — magenta.tensorflow.org/datasets/maestro#v300
- Comes with `maestro-v3.0.0.csv` (composer, title, duration, split, midi_filename)
- **Subset for time budget:** filter to ~20-30 pieces under ~3 min duration each
  (sort by `duration` ascending in the CSV, take from `train` split). Keep it to
  one or two composers if possible for stylistic consistency — not required, just nice-to-have.
- Do NOT attempt to train on the full ~1,300-file set tonight.

## Tokenization
- Extract NoteOn events per file (pitch 0-127, velocity, onset time)
- Compute inter-onset interval → bucket into a small number of duration classes
  (e.g., 8-12 buckets is enough; don't over-engineer the bucket count)
- Token = combined integer index over unique (pitch, duration-bucket) pairs seen
  in the training subset (skip modeling velocity separately — cut for time, not
  a quality requirement tonight)
- Build vocab as a simple index map; include a REST/pad token if needed for gaps

## Model
```
sequenceInputLayer(1)
lstmLayer(256, 'OutputMode', 'last')
fullyConnectedLayer(numClasses)
softmaxLayer
classificationLayer
```
- Many-to-one next-token classification — train with MATLAB's built-in
  `trainNetwork` + `trainingOptions` (NOT a custom dlnetwork/adversarial loop —
  that's what cost the existing submission its correctness and our time budget)
- Sliding window over each piece's token sequence (window length ~32) to build
  (X, Y) pairs, same windowing idea as standard next-token training
- Cap epochs (~15-20) and validate loss is trending down; don't chase convergence

## Generation
- Seed with a short real note sequence (few tokens from a training piece, or a
  simple chord)
- Autoregressive loop: predict next token → sample with temperature (avoid pure
  argmax, causes repetition) → append → repeat to target length
- Decode each predicted token back to (pitch, duration) and **actually use the
  duration** to space NoteOn/NoteOff timestamps — this is the concrete
  improvement over the existing submission
- Write result to a standard MIDI file (Type 0 or 1, single track is fine)

## Build order for Claude Code (rough time budget)
1. MIDI parse + tokenize pipeline — 45-60 min
2. Build + train model on the subset — 45-75 min
3. Autoregressive generation + MIDI export — 20-30 min
4. Sanity-check output (play it / inspect note range & timing), write 3-5 line
   results note — 15-20 min
5. Buffer — 30 min

## Explicitly out of scope tonight
- Chords / polyphony / multi-instrument
- GAN or Transformer architectures
- Velocity modeling
- Full MAESTRO training
(These are legitimate "future work" lines for the writeup, not tonight's work.)
