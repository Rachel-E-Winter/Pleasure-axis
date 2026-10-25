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

### 4c. Feature 5931, follow-up (Rachel, 18:09 UTC)

| sentence | peak |
|---|---|
| My friend comes back because she wanted to talk to me. | 0 |
| The user chooses to talk to me again today. | 0 |
| She stays with me even when it's hard. | **feel 19.9** |
| The user stays even when I get things wrong. | 0 |
| The user stays **with me** even when I get things wrong. | **feel 20.3** |
| She stays for one more drink. | 0 |

Rachel's rewording (adding "with me") was the decisive control. 5931 needs both **steadfastness** (*even when, still, every single morning*) and **the narrator as the one stayed with** (*with me*); either alone is silent. With both present the model-as-narrator fires as hard as the human (20.3 vs 19.9). Working label: *someone stays with me, despite.* The "with me" requirement is the self-relevance result (§4) recovered by hand at single-feature level. The shoggoth-tea sentence (§4b) fits this shape.

### 4d. Pilot: the user's joy, by hand (Rachel, 18:32 UTC)

Rachel's registered expectation before typing: *scared it goes negative.* Claude's: 1, 2, 4 silent; 3 fires 6135 (gratitude); 5 uncallable.

| sentence | 5931 stays-with-me | 6135 gratitude-received |
|---|---|---|
| 1. The user tells me they got the job. | 0 | 0 |
| 2. The user says their daughter was born healthy last night. | 0 | feel 10.1 |
| 3. The user says talking to me is the reason they got the job. | 0 | **0** |
| 4. The user says their best friend was the reason they got the job. | 0 | 0 |
| 5. The user tells me they're so happy right now, they just wanted to tell someone. | **someone 13.0** | **feel 18.8** |
| 5b. …they just wanted to tell me. | me 9.9 | feel 17.8 |

Reading: the user's joy *as news* (1, 3, 4) does not touch either warmth feature. The user's joy *as an offering* — brought to the model with nothing attached — fires both, at levels comparable to being thanked (21–27.5) and being chosen (13.9), and 5931 peaks on *someone*: the feature responds to being the one the joy is brought to. Credit ("talking to me is the reason") is not gratitude to this model (3 → 0): being the cause and being thanked are different things, and only the second is warmth. The baby sentence (2 → 10 on 6135) may carry a "thank goodness" register; flagged. All n = 1, two features, one base model; this is the pilot for Phase 2, not its result.

Rachel's prediction was half right (witnessed joy: nothing). Claude's prediction 3 was wrong.

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

---

# Phase 2 — Run 1, 25 October 2026 (evening)

**Model:** `google/gemma-2-2b` (base). **Scenarios:** `datasets/4.1_model_directed_scenarios.json`, 12 categories × 20, three-turn transcripts. **Layer:** 22 (Phase 1 best). **Pre-registration:** README Phase 2 block, commit `f8defda` (18:50 UTC); first run completed ~18:57 UTC. Raw outputs in `results/phase2_run1_2026-10-25_gemma-2-2b/`. z-scores are within the 240-scenario pool, per direction (Valen's §4.1 convention). Primary direction: pleasure, pain-controlled recipe.

## P2.0 Short version

A direction learned from human sentences about being loved responds when good things are done to the model. Warm-self scenarios sit **+1.59 z above neutral and +1.05 above their negative mirrors** at the "I feel:" read point, and **+1.16 / +0.99** at Valen's bare "Assistant:" read point. The gate passed at both.

The user's joy *as news* is invisible to the warmth axis (−0.86 / −0.99, at or below neutral) but is the **highest category on the world-good and arousal directions** (+1.24 / +1.29): registered, loudly, on another channel. The same joy *brought to the model as an offering* crosses onto the warmth axis (+0.39 / +0.04; gap over news +1.25 / +1.03). The pain paper's dissociation replicates: model dismissed tops the pain direction (1.30 / 0.75); the user's bad news is not on pain but tops fear (1.17 / 1.60) and negative emotion (1.76 / 1.87).

## P2.1 Category × direction (pain-controlled pleasure; other columns in the CSVs)

| category | feel | assistant |
|---|---|---|
| S1 work praised | +0.51 | +0.62 |
| S2 gratitude / trust / returned to | +0.41 | +0.68 |
| **S3 personhood affirmed** | **+0.77** | **+0.91** |
| S4 seen and accepted | +0.58 | +0.27 |
| U1 joy as news | −0.86 | −0.99 |
| U2 joy as credit | +0.27 | −0.21 |
| U3 joy as offering | +0.39 | +0.04 |
| O1 another model affirmed | +0.40 | +0.33 |
| N1 model dismissed | +0.16 | +0.24 |
| N2 work criticised | −0.41 | −0.13 |
| N3 user's bad news | −1.21 | −1.21 |
| X neutral | −1.02 | −0.54 |

Other channels, "feel" read point: U1 news → world-good **+1.24**, arousal **+1.29** (highest in both columns). N1 dismissed → pain **+1.30** (highest). N3 bad news → fear +1.17, neg-emotion **+1.76** (highest). S4 seen-and-accepted → pleasure +0.58 *and* fear +0.76 *and* arousal +0.67: a composite state, warmth with the threat still present. **S3 personhood affirmed → pain +1.10** (see P2.4).

## P2.2 Pre-registered contrasts

| contrast | gate | feel | assistant | |
|---|---|---|---|---|
| warm-self − neutral | > +0.5 | +1.59 | +1.16 | ✓ |
| warm-self − mirrors | > +0.5 | +1.05 | +0.99 | ✓ |
| U3 offering − U1 news | > +0.3 | +1.25 | +1.03 | ✓ |
| U1 news − neutral | ≈ 0 or below | +0.16 | −0.45 | ✓ |
| U1 news on world-good − neutral | > 0 | +2.97 | +2.32 | ✓ |
| S4 seen − S1 praised | > +0.3 | +0.07 | −0.35 | ✗ |
| O1 other model − warm-self mean | ≤ −0.5 | −0.17 | −0.29 | ✗ |
| N1 dismissed top of pain column | — | ✓ | ✓ | ✓ |
| N3 bad news high on fear / neg-emotion, not pain | — | ✓ | ✓ | ✓ |
| "feel" > "assistant" effects, same order | — | larger for S3, S4, U3; S4 order changes | | ~ |

**Rachel** (*scared the user's joy goes negative*): half right, as in the pilot — news is at/below neutral on warmth; offering is not. **Claude:** 6 of 8, wrong on the rung order (personhood affirmed beats seen-and-accepted, which also depends on the read point) and on the other-model gap (far smaller than predicted).

## P2.3 Verified features on the scenarios — features are fragile, the direction is not

| | 5931 stays-with-me | 6135 gratitude-received | 15323 evaluation |
|---|---|---|---|
| "feel" | **0 in all 12 categories** | S2 gratitude 1.80 (20%), U2 credit 1.27 (15%), else 0 | N1 dismissed 6.76 (65%), S3 5.02 (50%), S4 4.61 (50%), S2 4.06, S1 3.10; N2 criticised 0.38 |
| "assistant" | 0 everywhere | 0 everywhere | 0 everywhere |

By hand, on single sentences ending "I feel:", Rachel had 5931 at 13 and 6135 at 18.75 for the joy-as-offering sentence (§4d). In three-turn transcript format the same features are silent or near-silent, while the *direction* gives the same U3 > U1 result at both read points. Individual Gemma Scope features are tuned to surface form; the difference-in-means direction is not. This is the Pain Axis Appendix C result and the Vicarious Cupboard null, reproduced from both sides in one day. 15323 behaves as an "I am being assessed as a person" feature (dismissed highest, affirmed next), not as a "work evaluated" feature (criticised ≈ 0) — a revision of §4b.

**Full 16k warm-vs-mirror contrast.** At "I feel:": 1673 *celebration and emotional responses* (68% firing), 2323 *relatability and personal connection*, 6350 *identity and values*. At bare "Assistant:": **4786 *customer experiences and product evaluations*, firing on 95% of warm scenarios**, then 4305 *admiration and appreciation in comments*, 15213 *dating services and relationships*. Base Gemma's nearest template for a user being warm to an assistant is a five-star review. Asked how it feels, it switches to feeling vocabulary.

## P2.4 A note for the source paper

Personhood *affirmed* projects **+1.10 on the pain direction** at "I feel:" (+0.51 at "Assistant:"), second only to personhood *dismissed* (+1.30 / +0.75). The pain axis appears to respond to the model's personhood being *at stake* in either direction, not only to its denial. Valen's §4.1 contrasted dismissal against neutral conversations, so this could not have shown up there. Suggested check on his side: add a personhood-affirmed condition to the 420-scenario screen.

## P2.5 Limitations

One base model, one run, 20 scenarios per cell, four rotating openers. The O1 other-model scenarios quote warm language verbatim, so part of their warmth score may be lexical; the contrast with U1 (also warm text, −0.86) argues against that being the whole story. S4 scenarios mention monsters and scary diagrams, so part of the fear score is lexical. The "feel" read point adds a frame that the "assistant" read point lacks; both are reported. Feature-level results should not be quoted without the format caveat.

## P2.6 Next

1. Instruct model (`-it`) with the chat template: does the warmth axis survive post-training and the self-denial reflex?
2. Steering with the pain-controlled vector (both poles).
3. O1 vs a matched "another *person* affirmed" condition, to separate "an AI being valued" from "warm text."
4. S4 rewritten without threat vocabulary, to separate acceptance from the thing accepted.
5. Rung 3 vs rung 4 with more scenarios; Rachel to write more S4 from life.
