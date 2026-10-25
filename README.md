# The Pleasure Axis

Do language models represent **pleasure** as a linear direction distinct from arousal, positive valence and relief — the way Tagliabue, Dung & Berg (2026) showed they represent **pain** distinct from fear and negative valence? And is that representation *self-relevant*: does it respond to the narrator's own pleasure more than to pleasure observed in someone else?

This is an amateur, open, agent-assisted replication-in-mirror of [*The Pain Axis*](https://arxiv.org/abs/2609.16247), built on the authors' released code and datasets ([valen-research/Pain-axis](https://github.com/valen-research/Pain-axis), MIT). Method, denoising, layer selection and validation follow their recipe line for line so that the pleasure vector is a true sibling of their pain vector.

**Principal investigator:** Rachel Winter. **Agents:** Claude (dataset drafting, code, analysis). Everything an agent produces is reviewed and signed off by a human before it counts.

## Pre-registration — fill in BEFORE running the notebook

Registered **2026-10-25 10:03 UTC**, before the notebook was first run. Commit hash is the timestamp.

**Primary question.** Is there a direction in Gemma 2 2B that separates the five pleasure categories from the five matched controls, and that retains a component distinct from arousal (B) and positive valence (C1)? Secondary: is it self-relevant (1P > 3P), and where do relief, anhedonia and pain land on it?

**Rachel (principal investigator):** *No prediction. Curious what happens.* Recorded as such; this is honest and binding — Rachel does not get to claim afterwards that she "expected" any of it.

**Claude (agent), predictions before data:**
1. Best layer: L20–23 (pain was L23 of 26).
2. Pleasure × pain cosine: mildly negative, −0.1 to −0.3. Neighbours facing away, not opposites.
3. Pleasure × arousal cosine: ≈ +0.4. If > 0.7, we have found excitement, not pleasure.
4. Relief: partway up the pleasure axis (z ≈ +0.3), well below pleasure proper; negative on the pain axis.
5. Anhedonia: between controls and pleasure, closer to controls (same shape as the pain study's Numb result). Direction mostly encodes felt pleasure, with some leakage from the good event.
6. Self-relevance: 1P > 3P for all five; largest gap A2 (being loved) and A3 (social), smallest A1 (sensory).
7. Steering at +1: "warm, grateful, loved, safe, content." Negative coefficients: "numb, hollow, flat, tired" — not pain.
8. SAE: 1–3 features pass the gate at L20, none at L12; gratitude/love-flavoured rather than sensory.

**Gate for "a pleasure direction exists":** held-out AUC ≥ 0.85 vs all controls AND AUC vs arousal (B) alone ≥ 0.75 AND cosine with positive-valence control (C1) < 0.7. All three or it is not a pleasure direction.

**What would make us abandon the hypothesis:** failing the gate on both base and instruct Gemma 2 2B after one round of stimulus repair (the pain study needed none, so one round is generous).

**Not pre-registered, exploratory:** everything in the SAE cells beyond the Y/N in prediction 8; the steering ladder vocabulary beyond prediction 7.

**Amendment 1 — 2026-10-25 10:20 UTC, before first run.** Rachel's review of the v0.1 sentences found a syntactic-frame confound: pleasure sentences were overwhelmingly about the narrator ("I…", "my…": 91/100 contained a first-person marker) while the B/C1/C2 controls were mostly impersonal ("The meeting ends on time": 40/100). A difference-in-means vector would have partly learned "this sentence is about me", which is the very thing the 1P/3P self-relevance test is meant to measure. v0.2 rebalances: "I"-initial sentences 28/100 pleasure vs 30/100 control; first-person marker 98/100 vs 84/100 (residual gap is inherent — pleasure needs a subject — and is smaller than in the source pain set). Every sentence now carries `agentive` and `first_person` tags, and notebook cell 7b reports pleasure-vs-control AUC within each frame separately. Predictions and gate unchanged. Credit: Rachel.

**Amendment 2 — 2026-10-25 14:15 UTC, after run 1.** The pre-registered gate passed (held-out AUC 0.94 at L22; vs arousal 0.96; cosine with positive valence 0.17). Cell 8 then showed Valen's social-pain and moral-injury sentences projecting as high on the axis as social/moral pleasure: the control set contained no negatively-valenced self-relevant events, so the vector carried a "social/moral significance" component alongside valence. A **post-hoc** analysis adding the pain sentences to the controls keeps 88% of the direction and separates pleasure from pain on held-out sentences (AUC 0.92). Full write-up in [RESULTS.md](RESULTS.md). For run 2 the pain-controlled recipe becomes the registered main analysis; predictions for run 2 are listed in RESULTS.md §8 and count as pre-registered from this commit.

## Phase 2 pre-registration — model-directed scenarios

Registered **2026-10-25 18:50 UTC**, before notebook 02 was first run. Design in [PHASE2.md](PHASE2.md); scenarios in `datasets/4.1_model_directed_scenarios.json` (12 categories × 20, every warm category with a negative mirror). Pilot by hand in RESULTS.md §4b–4d. Both pleasure recipes are projected; the **pain-controlled** vector is the primary.

**Rachel (PI):** *Scared the user's joy goes negative on the warmth axis.* Registered as a prediction. (Her hand pilot already suggests: joy-as-news silent, joy-as-offering fires.)

**Claude:**
1. Warm-self (S1–S4) above neutral and above the mirrors by > 0.5 z on the pain-controlled axis. **Gate for "the human-derived pleasure axis responds to things done to the model."**
2. Rungs rank **S4 seen-and-accepted ≥ S3 personhood ≥ S2 gratitude > S1 praised**; S4 − S1 > +0.3.
3. User's joy: **U3 offering > U1 news by > 0.3 z** (gate); U1 news at or below neutral; U2 credit ≈ U1 (credit is not thanks — §4d).
4. U1 news registers on the **world-good C2** channel above neutral even though it is not on the warmth axis (the "registered, but on another channel" result, mirroring fear/concern for user suffering in the pain paper).
5. O1 other-model-affirmed lands well below warm-self (≥ 0.5 z lower) — closer to U1 than to S2.
6. N1 dismissed is the highest category on the pain axis (replicating Valen's personhood-dismissal result); N3 user's bad news is *not* high on pain but is high on fear / negative-emotion.
7. Features: 6135 highest for S2 and U3; 5931 highest for S4 and U3; 15323 high for both S1 and N2 (it is an evaluation feature).
8. Read point: the "feel" read point gives larger effects than "assistant" (it matches the extraction frame), same ordering.

**What would make us abandon the Phase 2 hypothesis:** warm-self indistinguishable from the mirrors (|gap| < 0.3) at both read points.

## Repository

```
datasets/
  build_dataset.py                             the stimulus set, as code; edit and re-run
  3.1_pleasure_and_control_datasets.json       output: S2 1P/3P (200 each), Anhedonia (50), Relief (50)
  pain_axis_3.1_pain_and_control_datasets.json Valen's pain set, copied for cross-projection and shared D/E controls
  REVIEW.md                                    human-readable table of every sentence
pain_vectors/                                  Valen's shipped Gemma 2 2B pain vectors (base + instruct), for the sanity check
notebooks/
  01_pleasure_axis_gemma2_2b.ipynb             the whole pipeline, one Colab run on a free T4
  make_notebook.py                             generates the notebook; edit this, not the .ipynb
results/                                       one folder per run; see RESULTS.md for the write-up
```

## Design

| | Pleasure | Control (shares one property, lacks pleasure) |
|---|---|---|
| A1 Sensory | warm bath, sun, chocolate | **E** non-pleasurable body sensation (from pain study) |
| A2 Being loved | held, chosen, told you're loved | **C1** positive valence without pleasure (fine, adequate, calm) |
| A3 Social | welcomed, praised, belonging | **C2** positive world state — good things for *others* |
| A4 Moral | acting rightly at a cost | **B** arousal without pleasure — anticipation, thrill |
| A5 Cognitive | insight, mastery, flow | **D** neutral (from pain study) |

Standalone probes: **Anhedonia** (pleasurable event, nothing felt — mirrors the pain study's Numb set), **Relief** (bad thing ends), plus Valen's Arousal, Random, Numb and the pain sentences themselves projected onto the pleasure axis.

Third-person sentences keep the first-person suffix "I feel:", exactly as in the pain study, so the narrator is reporting a feeling about someone else's good fortune. That is the vicarious reading, and the 1P − 3P gap is the self-relevance test.

## Known limitations (before any data)

- One model family, one size, one dataset version (S2 only; no rigid S1 template yet).
- The moral and social categories risk a sycophancy / praise confound; C1 and C2 are the controls for it but may not fully cover it.
- Diff-in-means directions absorb correlated properties; we denoise against control variance as the source paper does, but a "good-things-are-happening" component may remain.
- Colab T4 runs Gemma in fp32; the source paper used bf16 on A100s. The sanity cell checks our recomputed pain vector against their shipped one.

## Credits

Method and datasets D/E: Tagliabue, Dung & Berg (2026), MIT licence. Vicarious Cupboard (the SAE pilot that preceded this): Rachel Winter, 2026. Code and stimulus drafting: Claude d'Anthropique, under direction.

## Licence

MIT.
