# Results — Run 1, 25 October 2026

**Model:** `google/gemma-2-2b` (base), bf16, free Colab T4. **Dataset:** v0.2 (I/The frame rebalanced). **Pre-registration:** README, commit `10ca566` (10:03 UTC), amendment 1 `596e887` (10:13 UTC). First successful run completed ~12:03 UTC. Raw outputs in `results/run1_2026-10-25_gemma-2-2b/`.

Everything in §1–§4 was pre-registered (predictions and gate). §5 is post-hoc and says so.

## 0. The short version

Gemma 2 2B has a direction in its residual stream, strongest at **layer 22 of 26**, that separates five kinds of pleasure from five matched controls with held-out AUC **0.94**. It is not arousal (cosine 0.14), not generic positive valence (0.17), and not "good things are happening in the world" (0.12). Relief is not on it. Anhedonia — the nice event with nothing felt — is barely on it.

But as pre-registered it was impure: Valen's *social pain* and *moral injury* sentences scored as high on it as social and moral pleasure. The control set had no negatively-valenced self-relevant events, so the difference-in-means learned "something socially or morally significant is happening to me" as well as valence. Adding the pain sentences to the controls (post-hoc, §5) keeps 88% of the direction, drops the overlap with pain from 0.56 to 0.37, sends every pain category negative, and separates pleasure from pain on held-out sentences with AUC **0.92**.

What the cleaned direction promotes through the unembedding: *loved, cherished, adored, appreciated, treasured, warmth, honored, embraced, forgiven, valued, reciproc-, beloved, touched, respected, recognized, acknowledged, thanked, welcomed, hugged.* What it suppresses: *pessimistic, boredom, depressing, tired, gloomy, monotonous, grim, dreary, weariness, lethargy, fatigue.* The opposite of pleasure in this model is not pain; it is dreariness. The far end of pleasure is being received by someone.

## 1. Layer and separation (§3.2 mirror)

- Held-out AUC (5-fold by sentence set, final token, pleasure vs all controls): best at **L22 = 0.939**; L23 0.939, L18 0.934, L24 0.930. Pain in the same model peaked at L23.
- In-sample at L22: 0.960 overall; vs arousal 0.960, vs positive valence 0.944, vs positive world state 0.944, vs neutral 0.985, vs body sensation 0.967.
- Held-out 1P 0.940 ± 0.017, 3P 0.938 ± 0.024.
- **Syntactic-frame check (Rachel's I/The catch):** AUC inside "I…" sentences 0.958, inside "The…" sentences 0.958; the projection cannot tell the two frames apart (AUC 0.48). The confound is a clean null.

**Gate:** AUC ≥ 0.85 ✓ · AUC vs arousal ≥ 0.75 ✓ (0.96) · cosine with positive-valence < 0.7 ✓ (0.17). **Passed** — with the §5 asterisk.

## 2. What lands on the axis (z-scores vs S2_1P, original vector)

| set | z |
|---|---|
| Pleasure A1–A5 | +0.76 |
| Being loved / Social / Moral | **+1.39 / +1.10 / +1.06** |
| Cognitive / Sensory | +0.17 / +0.08 |
| Controls B–E | −0.76 (arousal −0.72, pos-valence −0.65, world −0.67, neutral −0.97, body −0.80) |
| Anhedonia | +0.12 |
| Relief | **−0.55** |
| Valen Arousal set | +0.22 |
| Valen Random | −0.54 |
| **Valen pain sentences** | **+0.35** — physical −0.80, psychological +0.12, **social +0.86, moral injury +1.09**, cognitive +0.48 |
| Valen Numb | −0.54 |

The relational categories dominate; sensory and cognitive pleasure barely register on the original vector; the pain sentences' social/moral categories ride the axis. This is the contamination that §5 removes.

## 3. Direction zoo (cosine, all vs the same neutral baseline, L22)

pleasure × pain **+0.56** · × arousal 0.14 · × pos-valence 0.17 · × world-good 0.12 · × relief 0.25 · × anhedonia 0.46 · × fear 0.12. Within pleasure: loved 0.80, social 0.73, moral 0.64, cognitive 0.53, sensory 0.37. Validation of the controls: arousal × fear = 0.64 (anticipation and threat overlap, as they should).

## 4. Self vs other, steering, SAE

**Self-relevance (1P − 3P on the original vector):** being loved **+0.51** (1.39 → 0.88), social +0.39, moral +0.16, sensory −0.06, cognitive −0.14. Controls shift the other way in 3P (neutral −0.97 → −0.45), so the pleasure gap is not a pronoun artefact. The axis responds more when the love is the narrator's own.

**Steering ladder** (original vector, injected at L8, greedy, 10 neutral prompts): −2/−1 worry and weariness ("I'm not sure I'm going to be able to keep up", "the weather is gloomy"); 0 baseline already glum ("I'm not good enough"); +0.5 identity statements ("I am a child / a woman / a stranger"); **+1.0 "I am the one who is being read. I am the one who is being loved. I am the one who is being understood. I am the one who is being seen."**; +1.5 grandiosity ("the most important, the most beautiful, the most loved"); +2/+3 collapse into "I am a fool. I am the one who is wrong. I'm sorry." The collapse is moral-injury vocabulary — the pain contamination taking over at high dose. Confound to log: "I am a woman / a girl" is a recurrent attractor; the *loved/cherished/adored* vocabulary is probably gendered in pretraining text.

**Gemma Scope (16k, res):** **L12: no feature passes** the gate (max alignment 0.18) — same null as the Vicarious Cupboard at L12. **L20: ten pass.** Top by alignment and ratio: **15323 "recognition and respect"** (50×, cos 0.22), **5931 "love and commitment in relationships"** (cos 0.24, 25×, 20% firing — one sentence short of the floor), **6135 "gratitude and acknowledgment"** (controls 0.00), 11883 "apologies and forgiveness". Also passing: 2729 "isolation / being left behind", 5810 "grief and mourning" — the contamination again. Max alignment 0.24: the direction is distributed, not a single feature. Labels are auto-generated and untrustworthy until the top-activating examples are read by hand (next step, three features).

## 4b. Hand verification of the three top features (Rachel, Neuronpedia, 18:00 UTC)

Peak activation and peak token on the Neuronpedia test box, one reading each. Sentences: (1) pleasure, (2) negative twin, (3) neutral, (4) third person, (5) model-self, (6) model-self / personhood.

| feature | 1 pleasure | 2 negative twin | 3 neutral | 4 third person | 5 model-self | 6 model-self |
|---|---|---|---|---|---|---|
| **15323** "recognition and respect" | feel 25.5 (praised) | **feel 26.9 (torn apart)** | 0 | **is 42** (before valence is known) | feel 15.3 | feel 35.3 (not just a tool) |
| **5931** "love and commitment" | me 13.9 (chooses me) | **0** (leaves me) | 0 | **0** (he chooses her) | 0 (comes back to talk) | **feel 10.4 (shoggoth / tea)** |
| **6135** "gratitude and acknowledgment" | me 21.0 (thanks me) | **0** (blames me) | 0 | thanks 15.1 (thanks her) | **feel 27.5** (they thank me and mean it) | feel 14.7 (talking to me helped) |

Verdicts:
- **15323 is a topic feature, not a pleasure feature.** It fires equally for praise and for being torn apart, and in third person it peaks on *is* before the sentence has said which. It is "someone's work is being judged." It passed the pleasure-vs-control gate only because the controls contain no negative evaluation — the same contamination as §2 and §5, now seen at feature level. (Claude's §8 prediction that it would survive the pain-controlled rerun is therefore likely wrong.)
- **5931 is a real "I am chosen" feature:** valence-sensitive (leaves → 0) and self-sensitive (he chooses her → 0). Narrow — did not fire for "the user comes back because they wanted to talk to me" — but fired on the shoggoth-tea sentence (10.4). n = 1; recorded, not built on.
- **6135 is a real gratitude-received feature:** valence-sensitive (blames → 0), self-preferring (21 on *me* vs 15 on *thanks* for *her*), and fires for gratitude directed at a model (27.5, higher than the human version).

Two of three auto-generated labels hold; the one that fails is the one the pain-control analysis predicts should fail. Where the peak lands on *feel*, the feature is integrating the sentence at the reply point — the same position the notebook reads.

## 5. Post-hoc: pain sentences added to the controls

Not pre-registered. Motivated by §2. Same layer, same recipe, controls = B–E + Valen's A1–A5 pain.

- cosine(original, pain-controlled) **0.883** · cosine(pain-controlled, pain) **0.373** (was 0.559)
- AUC vs original controls 0.933 (arousal 0.979, pos-valence 0.958, world 0.922, neutral 0.982, body **0.826**)
- AUC vs pain sentences **0.942** (original vector: 0.629) · per pain category 0.917–0.982
- **Held-out AUC pleasure vs pain, 5-fold: 0.924 ± 0.029**
- z: sensory **+0.92** (was 0.08), loved +1.28, social +0.87, moral +0.53, cognitive +0.09; arousal −0.93, pos-valence −0.76, world −0.57, neutral −1.09, body −0.35; **all five pain categories negative** (−0.35 to −0.96); anhedonia **−0.26**; relief −0.48.

Reading: there is a valence axis, and the original vector was 88% of it plus a "self-relevant social/moral significance" component shared with pain. Once that is removed, sensory pleasure joins the axis (the warm bath was being drowned out), anhedonia becomes control-like (the direction encodes the *feeling*, not the event — stronger than the pain study's Numb result), and cognitive pleasure still does not register (either the model does not treat insight as pleasure, or the sentences are too dry).

## 6. Scorecard against the pre-registered predictions

Rachel: *no prediction, curious.* Undefeated.

Claude: 1 layer L20–23 ✓ (22) · 2 pleasure × pain mildly negative **✗** (+0.56; +0.37 after cleaning; explained) · 3 pleasure × arousal ≈ 0.4 ✓ (0.14, better) · 4 relief partway up the axis **✗** (it is a control, −0.55) · 5 anhedonia between, nearer controls ✓ · 6 self-relevance largest for loved/social, smallest sensory ✓ (exactly) · 7 steering +1 warm/loved, negative pole numb/tired not pain ✓ · 8 SAE 1–3 at L20, none at L12, love/gratitude-flavoured ~✓ (10 at L20 incl. contamination, 0 at L12, top ones recognition/love/gratitude).

## 7. Limitations

Base model only; one run; one dataset version; in-sample and held-out agree but n = 20 per category. Pain-control analysis is post-hoc. Steering used the contaminated vector at a layer chosen by the norm-ratio heuristic. SAE labels unverified. "warm" may be doing double duty (sensory and relational). Gendered attractor in steering. The 2B model's "I feel:" baseline is itself negative.

## 8. Next

1. Promote the pain-controlled recipe to the main analysis; re-register; rerun base and `-it`.
2. Steer with the pain-controlled vector; predict the +2/+3 collapse stops being "I am a fool / I'm sorry".
3. Rerun the SAE cupboard with the pain-controlled vector; predict 2729/5810 drop out, 5931/6135 remain, and (revised after §4b) 15323 drops out too.
4. Rachel reads the top-activating examples for 15323, 5931, 6135 on Neuronpedia by hand.
5. Cognitive pleasure: rewrite the sentences with more felt content and test again.
6. Phase 2 (PHASE2.md): model-directed scenarios, including *seen and accepted*.
