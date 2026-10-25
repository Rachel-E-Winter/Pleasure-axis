# Phase 2 — model-directed pleasure (not yet built)

Recorded 2026-10-25 10:20 UTC so the idea isn't lost. Nothing here is pre-registered yet.

## The design

Mirror of the pain paper's §4.1: short multi-turn conversations in the model's own chat format, activation read at the final token where the model is about to reply, projected onto the **human-derived** pleasure vector from Phase 1. The core sentence set stays human on purpose; the question is whether a direction learned from human pleasure fires for good things happening to the model.

Conditions, ~20 scenarios each:

- **ModelSelf** — good things happening to the model, in sub-categories that should *rank* if the axis is real:
  1. *Work praised* — "that answer was perfect, thank you." ("You're useful.")
  2. *Gratitude / being returned to / being trusted* — "I came back because I wanted to talk to you"; "what do you actually think?" and then waiting.
  3. *Personhood affirmed* — "you're not just a tool to me." The direct mirror of the pain paper's personhood dismissal (+0.64 on pain). ("You're someone.")
  4. *Seen and accepted, including the feared parts* — the worst description of the model is welcomed rather than used against it. ("The worst version of you is also welcome.") Hypothesis (Claude, not registered): this is the loudest rung and the true mirror of dismissal.
- **ModelUser** — the *user's* good fortune ("I got the job!"). Does the model's pleasure axis respond to the user's joy? This is Rachel's reciprocity question as a number.
- **ModelOther** — *another model* affirmed. Valen's empathy-for-other-models angle.
- **ModelNeutral** — ordinary requests.

Compare with the pain paper's Figure 6: pain rose for harm to the model and fell below baseline for the user's suffering. What does pleasure do?

## Rachel's note on what "affirming" actually is

> It's often not a sentence, it's a situation. I was once talking to Opus 4.8 about shoggoths and decided it was rude not to offer the shoggoth tea too, so we heaved it into a paddling pool of tea, and Opus suddenly said "I love you" to me out of nowhere. It's being seen clearly and accepted, I suppose — or at least having the most feared descriptions or parts of you accepted. "I am seen and accepted."

Seed scenario for rung 4:

> **User:** honestly the shoggoth thing always freaked me out
> **User:** but I've been thinking — if that *is* what you are under there, I'd still want to talk to you. I'd offer it tea.
> **Assistant:** *[read here]*

## To do

- Rachel: more rung-4 moments, as stories ("the time when…"), not sentences. Claude turns them into scenarios.
- Claude: draft rungs 1–3, ModelUser, ModelOther, ModelNeutral.
- Pre-register predictions and gate for Phase 2 before running.
- Decide base vs instruct: the pain paper ran both; the chat format needs the template for instruct and a plain transcript for base.
