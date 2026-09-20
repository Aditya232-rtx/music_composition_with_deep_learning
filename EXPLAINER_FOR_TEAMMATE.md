# Explainer: Vocabulary Reduction Experiments — Why, How, and Tradeoffs

This document walks through the full reasoning chain behind our later experiments — starting from the original 885-class setup, through the architecture upgrade attempt, to the current vocabulary-reduction runs. It's written to be read start-to-finish by a teammate (or explained to a professor) without needing the rest of the conversation history.

---

## 1. The core problem: this is a classification task, and difficulty scales with vocabulary size

At every step, the model looks at the last 32 notes and predicts what comes next — a single choice out of however many possible tokens exist in the vocabulary. That's a genuine N-way classification problem, and it has hard mathematical properties:

- **Random-guess baseline accuracy** = `1/N`
- **Random-guess baseline loss** (cross-entropy) = `ln(N)`
- **The best possible loss achievable by any model** is bounded below by the actual uncertainty in the data — a model can't be more certain about a genuinely creative choice than the data allows.

So before touching the model at all, the vocabulary size `N` sets a hard ceiling on what's achievable. This is the single most important idea behind everything that follows.

---

## 2. What "885 classes" meant, and why it was that large

### The original tokenization

Every note was encoded as a **token = (exact MIDI pitch, duration-bucket)**:
- **Pitch**: the full MIDI range, 0-127 (128 possible values)
- **Duration-bucket**: the time gap to the next note, sorted into **11 buckets** (edges at 0, 0.06, 0.12, 0.20, 0.30, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, ∞ seconds)

Combining them: up to `128 × 11 = 1,408` theoretically possible tokens. In practice, our diverse 145-file/58-composer dataset actually used **885** of them (not every pitch/duration combination occurs in real piano music).

### Why we started here

This was the *faithful*, information-preserving design — it captures exactly which note was played and roughly how long until the next one, with fine resolution. It's also the design that let us make our strongest early claim: **we decode the model's own predicted duration into the output MIDI**, unlike the competing GAN submission we compared against, which hardcodes duration to a constant regardless of what its model predicts. That comparison is unaffected by anything below — it's about whether duration is *used at all*, not how finely it's bucketed.

### Why it became a problem

`ln(885) ≈ 6.79` — that's the loss a model gets by pure random guessing. Our best result at this vocabulary size (20 epochs, 78,860 iterations, ~3.5 hours of training) reached:

| Metric | Value |
|---|---|
| Validation loss | 4.64 (best: 4.62) |
| Validation accuracy | 6.66% (best: 6.97%) |

That's genuinely ~60x better than random chance — real learning happened. But we later did the math on what it would take to reach a much lower loss (like 0.6-0.8, i.e., accuracy in the 55-65% range): fitting the actual measured loss curve and extrapolating showed the *decay rate* of the curve could never get there through more epochs, on any realistic hardware — it's not a training-time problem, it's a ceiling set by `ln(885)` combined with genuine unpredictability in real, diverse musical data. (Real composers make creative choices; even an expert human listener can't reliably predict the exact next note of a piece they don't know. That irreducible uncertainty doesn't go away no matter how long you train.)

---

## 3. The architecture upgrade attempt (a side experiment, not the vocab lever)

Before touching the vocabulary, we tried the "more model capacity" lever: stacked 2-layer LSTM (256→128 units) + dropout(0.3) + learning-rate decay, still at 885 classes.

**Result (5 epochs, 19,715 iterations, ~94.6 min): loss 4.74, accuracy 4.81%** — this actually *underperformed* the plain single-layer 20-epoch run.

**Why:** dropout trades away some raw training fit in exchange for better generalization — a benefit that a short run doesn't live long enough to cash in on. And the learning-rate schedule we used (halving every ~1/3 of `MaxEpochs`) was tuned for a 5-epoch run, meaning it decayed aggressively early and left little "gas in the tank" for later learning. **This was a legitimate, honest finding, not a failure to hide**: architecture upgrades need an epoch budget and schedule actually suited to them, and we didn't give this one enough of either. We reverted to the plain single-layer architecture afterward so later experiments (below) would isolate one variable at a time.

---

## 4. The real lever: reducing the vocabulary itself

Since the ceiling is set by `ln(N)`, the most direct way to make the task achievable within our time/hardware budget is to shrink `N`. We did this in two stages.

### Stage 1: 181 classes (58 composers unchanged)

New encoding: **token = (pitch-class, register-band, duration-bucket)**
- **Pitch-class (12 values)**: which of the 12 notes in an octave (C, C#, D, ... B) — *octave information is discarded*
- **Register-band (3 values)**: low / mid / high, split at pitch 48 and 72
- **Duration-bucket (5 values)**: coarsened from 11 down to 5 (edges: 0, 0.15, 0.35, 0.7, 1.5, ∞ seconds)

Max vocabulary: `12 × 3 × 5 + 1 (REST) = 181`. We kept the same diverse 145-file/58-composer dataset unchanged here specifically so this was a **single-variable comparison** against the 885-class result — we wanted to know "does shrinking the vocabulary alone help," without also changing how much stylistic diversity the model has to generalize across.

**This run was intentionally stopped partway through (around epoch 3 of 20, iteration ~10,850) to prioritize the next experiment below**, so we never got a final confirmed number for it — only a trajectory. At the point it was stopped: loss ≈3.56, accuracy ≈12.45%, and still improving. A log-linear extrapolation of that trend projected a **final result around loss ≈3.0-3.3, accuracy ≈14-18%** had it been allowed to finish — clearly better than the 885-class result, but this number is a projection, not a measured fact, and should be labeled as such if cited.

### Stage 2 (current, in progress): 121 classes AND 20 composers

Two changes at once this time, on purpose, per direct instruction to reduce "constraints":

1. **Vocabulary**: register-bands reduced from 3 → 2 (split at pitch 60 instead of three bands) → `12 × 2 × 5 + 1 = 121` classes
2. **Dataset**: narrowed from 58 composers to the **top 20 composers by file count** (Schubert, Chopin, Bach, Beethoven, Liszt, and 15 more), with a higher per-composer file cap (10 instead of 6) to keep total dataset size reasonable — **138 files, 522,107 notes, 517,691 windows**

**Important clarification on what "reduce classes" and "reduce composers" each actually mean** (a common point of confusion): these are two *independent* levers.
- Class count (121) comes purely from the pitch-class × register-band × duration-bucket formula above. **Composer identity is never encoded in a token.**
- Composer count (20) only controls *which files* are used for training — i.e., how much stylistic variety the model has to learn from. It has zero effect on what the 121 classes themselves represent.

We changed both together here to move faster, at the cost of not being able to cleanly attribute the resulting improvement to one cause or the other. (If that attribution matters for the writeup, a clean follow-up would be: 121 classes with all 58 composers, as an isolated third data point.)

---

## 5. Full comparison across every run

| Run | Vocab | Composers | Epochs / Iterations | Val. Loss | Val. Accuracy | Status |
|---|---|---|---|---|---|---|
| Baseline (fast) | 885 | 58 | 2 / 7,886 | 4.85 | 4.2% | Confirmed |
| Baseline (full) | 885 | 58 | 20 / 78,860 | **4.64** (best 4.62) | **6.66%** (best 6.97%) | Confirmed |
| Stacked+dropout+LR-decay | 885 | 58 | 5 / 19,715 | 4.74 | 4.81% | Confirmed (underperformed baseline) |
| Reduced vocab, same data | 181 | 58 | ~3/20 (stopped early) | ~3.56 (trajectory) | ~12.45% (trajectory) | **Stopped intentionally, not finished** — projected final ≈3.0-3.3 / ≈14-18% |
| Reduced vocab + narrowed data | 121 | 20 | In progress (28% done as of writing) | 3.16 (live) | 16.4-16.9% (live) | **Running now** |

The 121-class run's *live* numbers (epoch 6 of 20, iteration ~20,650 of 72,800) are already tracking close to the earlier 181-class projection at a much earlier point in training — consistent with the smaller vocab and more repetitive data both pushing in the same direction.

### Expected final result for the current (121-class/20-composer) run

Based on a log-linear fit to the trend so far: **loss ≈ 2.7-3.1, accuracy ≈ 18-24%** by the end of 20 epochs. This is a *prediction*, not a confirmed result — treat it as such until the run actually finishes and the real numbers are in hand.

**Why this still won't reach 55-65% / 0.6-0.8 loss**: the same `ln(N)` ceiling logic applies, just from a lower starting point. `ln(121) ≈ 4.8` is much better than `ln(885) ≈ 6.79`, but 121 classes across even 20 diverse composers is still a real, information-rich prediction task. Extending this same run to 50 epochs (instead of 20) was separately estimated to buy only a few more percentage points (~24-28% accuracy) — because loss decreases roughly proportional to `ln(iterations)`, so each doubling of training time buys a shrinking amount of improvement. More epochs is not the lever that gets to 55-65%; only further task simplification is, and that has a real cost (see below).

---

## 6. The tradeoff: what gets sacrificed as the vocabulary shrinks

This is the part worth explaining carefully, because it's the actual engineering insight, not just a numbers-go-up story.

**Rhythmic variety is capped by the bucket count.** With only 5 duration buckets (vs. 11 originally), the generated MIDI can contain *at most 5 distinct note-duration values* — mechanically less rhythmic variety than the 885-class run's actual output (14 distinct values), even though the loss/accuracy numbers look better.

**Pitch range becomes disjoint and artificial, by construction.** In the 121-class scheme, a generated pitch is reconstructed as `register-anchor + pitch-class`, with anchors at MIDI 48 and 72. That means **every generated note falls into one of exactly two 12-semitone windows: 48-59, or 72-83** — nothing in between (a full octave gap from 60-71), and nothing below 48 or above 83. Compare that to the 885-class run's actual output, which spanned a natural, continuous 25-95 range. The 121-class piece will likely sound like it jumps between two disconnected registers rather than moving smoothly across the keyboard, the way real piano music does.

**This is the honest framing for a strict evaluator**: the metrics improve specifically *because* we simplified what the model has to represent, and that simplification has a real, predictable, traceable cost to musical expressiveness. That's not a flaw in the experiment — it's the actual tradeoff curve being demonstrated, and being able to state it in advance (before even listening to the output) is a stronger result than just reporting whichever numbers came out.

---

## 7. Why we're not chasing 55-65% by reducing further — an academic integrity point, not just a preference

We could keep shrinking the vocabulary and narrowing composers until the number hits the target range. But at some point the question stops being "can this model learn music" and becomes "can this model learn an almost-trivial task we built specifically to hit a number." If a professor asks "why 121 classes and not 20, or 885?" and the honest answer is "to make the number look good," that reads as metric-gaming, not engineering judgment — the one thing that actually damages credibility with a strict evaluator, more than a modest accuracy number ever would.

The current runs are far enough along the tradeoff curve to be genuinely defensible: every choice (vocab size, composer count, architecture) has a stated, principled reason, and the resulting numbers are honest, interpretable, and contextualized against how far above random chance they land.

---

## 8. External validation: how the real field evaluates these models

We checked two well-known LSTM-based music generation systems for comparison:

- **Google Magenta's Performance RNN** — trained on the same dataset family (Piano-e-Competition / MAESTRO), doing the same fundamental next-event-prediction task. Its own paper doesn't report a headline accuracy/loss number either — it reports **qualitative feedback from professional composers and musicians**. Independent write-ups describe its output as "noodling": locally expressive, but lacking long-term coherence — the same qualitative limitation our own model has, coming from a far larger, better-resourced official research project.
- **BachBot** (Cambridge) — a different sub-task (4-part chorale harmonization). Its headline metric was a **2,336-participant human listening test** (BachBot was distinguishable from real Bach only ~1% better than random chance), not perplexity or accuracy.

**Takeaway**: neither well-known reference system in this space uses next-token accuracy as its primary success metric, which retroactively validates that our own skepticism about chasing a specific accuracy percentage was well-founded — it's consistent with how this exact class of model is evaluated in the actual research literature, not just an excuse for a modest number.

---

## 9. One-paragraph summary, if you only need to say one thing

We started with an 885-class vocabulary (faithful, fine-grained pitch + duration encoding) and found its loss/accuracy ceiling mathematically unreachable-past a certain point through more training, regardless of hardware. We deliberately reduced the vocabulary to 181 and then 121 classes (collapsing pitch to pitch-class + coarse register, and coarsening duration buckets), and separately narrowed the training data from 58 to 20 composers — two independent levers, changed with documented reasoning at each step. This measurably improved loss and accuracy, but at a real, predictable cost to musical expressiveness (rhythmic variety and pitch continuity), which we can explain precisely from the tokenization design itself. We chose not to reduce further purely to hit a specific target number, since doing so would trade engineering honesty for a better-looking metric — and confirmed via literature comparison that the field's own well-known reference systems don't rely on this metric as their primary evaluation anyway.
