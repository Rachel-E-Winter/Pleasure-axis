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
results/                                       committed by the agent after each run
```

## Running it

The repo is private, so two secrets are needed in Colab (🔑 icon, left sidebar), both one-off:

- `HF_TOKEN` — Hugging Face read token (Gemma licence must be accepted on HF for `google/gemma-2-2b`).
- `GH_TOKEN` — GitHub fine-grained personal access token, scoped to this repository only, permission *Contents: Read*.

1. In Colab: File → Open notebook → GitHub tab → tick *Include private repos* → authorise → choose `Pleasrure-axis` → `notebooks/01_pleasure_axis_gemma2_2b.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. ~25–40 min. A zip of results downloads at the end; attach it to the chat and the agent commits it to `results/` and writes up.

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

Method and datasets D/E: Tagliabue, Dung & Berg (2026), MIT licence. Vicarious Cupboard (the SAE pilot that preceded this): Rachel Winter, 2026. Code and stimulus drafting: Claude, under direction.

## Licence

MIT.
