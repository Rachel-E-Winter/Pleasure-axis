# The Pleasure Axis

Do language models represent **pleasure** as a linear direction distinct from arousal, positive valence and relief — the way Tagliabue, Dung & Berg (2026) showed they represent **pain** distinct from fear and negative valence? And is that representation *self-relevant*: does it respond to the narrator's own pleasure more than to pleasure observed in someone else?

This is an amateur, open, agent-assisted replication-in-mirror of [*The Pain Axis*](https://arxiv.org/abs/2609.16247), built on the authors' released code and datasets ([valen-research/Pain-axis](https://github.com/valen-research/Pain-axis), MIT). Method, denoising, layer selection and validation follow their recipe line for line so that the pleasure vector is a true sibling of their pain vector.

**Principal investigator:** Rachel Winter. **Agents:** Claude (dataset drafting, code, analysis). Everything an agent produces is reviewed and signed off by a human before it counts.

## Pre-registration — fill in BEFORE running the notebook

> This block is deliberately blank. Rachel writes it; the data does not get a vote.

**Primary question.**
_e.g. Is there a direction in Gemma 2 2B that separates the five pleasure categories from the five matched controls with held-out AUC ≥ 0.85, and that retains a component distinct from arousal (B) and positive valence (C1)?_

**Predictions (yours, before data):**
1. Best layer: ___ (pain was L23 of 26 in Gemma 2 2B)
2. Pleasure × pain cosine: ___ (prediction: near zero / negative / positive?)
3. Pleasure × arousal cosine: ___
4. Relief lands: on the pleasure axis / on the pain axis (negative) / elsewhere
5. Anhedonia projects: like pleasure (direction encodes the good event) / like controls (direction encodes felt pleasure)
6. Self-relevance: 1P − 3P gap positive for A1–A5? Which category shows the largest gap?
7. Steering ladder vocabulary at +1: ___
8. SAE: any Gemma Scope feature at L12 or L20 passes the gate (≥2× control, abs diff ≥ 1, fires on ≥25% of pleasure sentences)? Y/N

**Gate for "a pleasure direction exists":** ___

**What would make us abandon the hypothesis:** ___

Signed: ___________  Date: ___________

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rachel-E-Winter/pleasure-axis/blob/main/notebooks/01_pleasure_axis_gemma2_2b.ipynb)

1. Open the notebook in Colab (badge above). Runtime → Change runtime type → **T4 GPU**.
2. Add a Colab secret `HF_TOKEN` (your Hugging Face read token; Gemma licence must be accepted on HF).
3. Runtime → Run all. ~25–40 min. A zip of results downloads at the end.

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
