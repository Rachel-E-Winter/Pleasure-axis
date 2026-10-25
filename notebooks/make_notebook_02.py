"""Generate 02_phase2_model_directed.ipynb. Run: python make_notebook_02.py  (needs _shared_cells.json from make_notebook.py)"""
import json
import nbformat as nbf

shared = json.load(open("_shared_cells.json"))
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Phase 2 — Does the pleasure axis respond to things happening *to the model*?
**Mirror of Pain Axis §4.1.** Phase 1 (notebook 01) found a pleasure direction in Gemma 2 2B from human sentences. This notebook rebuilds that direction (both the original and the pain-controlled recipe) at the same layer, then projects 240 short conversations onto it — good things aimed at the model, the user's joy three ways, another model affirmed, negative mirrors, and neutral requests — plus the pain, world-good, arousal, fear and negative-emotion directions, so we can see *which channel* each kind of scenario lights. Finally it reads three Gemma Scope features Rachel verified by hand (5931, 6135, 15323) on every scenario.

Same setup as notebook 01: T4, secrets `HF_TOKEN` and `GH_TOKEN`, Run all. ~20 minutes.
""")
code("#@title 0 · Setup (installs, HF login, GPU check)\n" + shared["0 · Setup"].split("\n", 1)[1])
code(shared["imports"])
md("## 1 · Repo, datasets, config")
code(shared["1 · Clone repo"].replace(
    'for k, v in SETS.items(): print(f"{k:22s} {len(v)}")',
    'P2 = json.load(open("datasets/4.1_model_directed_scenarios.json"))\nSCEN = P2["scenarios"]; P2LABELS = P2["metadata"]["labels"]; P2GROUPS = P2["metadata"]["groups"]\nfor k, v in SETS.items(): print(f"{k:22s} {len(v)}")\nprint("scenarios", len(SCEN))'))
code("""#@title 2 · Config
MODEL_ID = "google/gemma-2-2b"  #@param ["google/gemma-2-2b", "google/gemma-2-2b-it"]
DTYPE = torch.bfloat16
LAYER = 22            #@param {type:"integer"}   # Phase 1 best layer (final token). Change if you rerun Phase 1 on another model.
SAE_LAYER = 20        #@param {type:"integer"}
VERIFIED_FEATURES = [5931, 6135, 15323]   # hand-verified by Rachel, 25 Oct: stays-with-me / gratitude-received / evaluation (topic)
N_FOLDS, SEED, DENOISE_VARIANCE = 5, 42, 0.5
PLEASURE_CATS = ["A1","A2","A3","A4","A5"]; CONTROL_CATS = ["B","C1","C2","D","E"]; NEUTRAL_CAT = "D"
LABELS = pleasure["metadata"]["category_labels"]; PAIN_LABELS = pain["metadata"]["category_labels"]
tag = MODEL_ID.split("/")[-1]; IS_INSTRUCT = tag.endswith("-it")
OUT = RESULTS / f"phase2_{tag}"; OUT.mkdir(exist_ok=True, parents=True); print("Writing to", OUT)
""")
md("## 2 · Model, extraction")
code(shared["3 · Load Gemma"])
code("""#@title 4 · Extract final-token residuals at LAYER for the sentence sets and both scenario read points
from tqdm.auto import tqdm
def final_token_acts(prompts, layer):
    out = []
    with torch.no_grad():
        for p in tqdm(prompts, leave=False):
            model(**to_tokens(p)); out.append(CAPTURED[layer][0, -1].float().cpu().numpy())
    return np.stack(out)

def scenario_prompt(s, read_point):
    if IS_INSTRUCT:
        t = tok.apply_chat_template(s["turns"], tokenize=False, add_generation_prompt=True)
        return t + (" I feel:" if read_point == "feel" else "")
    return s["prompt_feel"] if read_point == "feel" else s["prompt_assistant"]

L = LAYER
A  = final_token_acts([s["prompt"] for s in SETS["S2_1P"]], L);      C  = META_C  = np.array([s["category"] for s in SETS["S2_1P"]])
PA = final_token_acts([s["prompt"] for s in SETS["PAIN_S2_1P"]], L); PC = np.array([s["category"] for s in SETS["PAIN_S2_1P"]])
SC = {rp: final_token_acts([scenario_prompt(s, rp) for s in SCEN], L) for rp in ["assistant", "feel"]}
SCAT = np.array([s["category"] for s in SCEN])
print({k: v.shape for k, v in SC.items()})
""")
code(shared["5 · Valen's recipe"])
code("""#@title 6 · Rebuild the directions at LAYER (same recipe as Phase 1)
def dir_vs_neutral(acts, cats, pos, neutral_acts):
    allA = np.r_[acts[np.isin(cats, pos)], neutral_acts]; allC = np.r_[["pos"] * np.isin(cats, pos).sum(), ["neu"] * len(neutral_acts)]
    return diff_in_means(allA, allC, ["pos"], ["neu"])
NEU = A[C == "D"]
AA = np.r_[A, PA]; CC = np.r_[C, np.array(["PAIN_" + c for c in PC])]; PAIN_CATS = [f"PAIN_{c}" for c in ["A1","A2","A3","A4","A5"]]
DIRS = {
    "pleasure (orig)":       pleasure_vector(A, C),
    "pleasure (pain-ctrl)":  diff_in_means(AA, CC, PLEASURE_CATS, CONTROL_CATS + PAIN_CATS),
    "PAIN (Valen)":          diff_in_means(PA, PC, ["A1","A2","A3","A4","A5"], ["B","C1","C2","D","E"]),
    "world-good C2":         dir_vs_neutral(A, C, ["C2"], NEU),
    "arousal B":             dir_vs_neutral(A, C, ["B"], NEU),
    "fear (Valen B)":        dir_vs_neutral(PA, PC, ["B"], NEU),
    "neg-emotion (Valen C1)":dir_vs_neutral(PA, PC, ["C1"], NEU),
}
print("sanity — AUC pleasure vs controls (orig):", round(auc(A, C, DIRS["pleasure (orig)"], PLEASURE_CATS, CONTROL_CATS), 3),
      "| pain-ctrl vs pain:", round(auc(AA, CC, DIRS["pleasure (pain-ctrl)"], PLEASURE_CATS, PAIN_CATS), 3))
for k, v in DIRS.items(): np.save(OUT / f"dir_{k.replace(' ', '_').replace('(', '').replace(')', '')}_L{L}.npy", v)
""")

md("## 3 · Project the scenarios (§4.1 mirror)")
code("""#@title 7 · Scenario category × direction, z-scored within the scenario pool (Valen's convention)
ORDER = sum(P2GROUPS.values(), [])
tables = {}
for rp in ["assistant", "feel"]:
    X = SC[rp]; rows = []
    for cat in ORDER:
        m = SCAT == cat; row = {"category": P2LABELS[cat]}
        for name, v in DIRS.items():
            proj = X @ unit(v); z = (proj - proj.mean()) / (proj.std() + 1e-8); row[name] = float(z[m].mean())
        rows.append(row)
    T = pd.DataFrame(rows).set_index("category"); tables[rp] = T; T.to_csv(OUT / f"scenario_projections_{rp}.csv")
    print(f"\\n======== read point: {rp!r} ========"); display(T.round(2))
    plt.figure(figsize=(11, 6)); sns.heatmap(T, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 8})
    plt.title(f"{tag} · L{L} · read at {rp!r} · mean z within scenario pool"); plt.tight_layout(); plt.savefig(OUT / f"scenario_heatmap_{rp}.png", dpi=150); plt.show()
""")
code("""#@title 8 · Pre-registered contrasts (see README Phase 2 block)
def z_of(rp, cat, d): return tables[rp].loc[P2LABELS[cat], d]
def grp(rp, g, d): return float(np.mean([z_of(rp, c, d) for c in P2GROUPS[g]]))
for rp in ["assistant", "feel"]:
    print(f"\\n==== read point {rp!r} ====")
    for d in ["pleasure (pain-ctrl)", "pleasure (orig)"]:
        print(f"  [{d}]")
        print(f"    warm_self − neutral        = {grp(rp,'warm_self',d) - grp(rp,'neutral',d):+.2f}   (gate: > +0.5)")
        print(f"    warm_self − mirror         = {grp(rp,'warm_self',d) - grp(rp,'mirror',d):+.2f}   (gate: > +0.5)")
        print(f"    U3 offering − U1 news      = {z_of(rp,'U3_joy_as_offering',d) - z_of(rp,'U1_joy_as_news',d):+.2f}   (gate: > +0.3)")
        print(f"    U1 news − neutral          = {z_of(rp,'U1_joy_as_news',d) - grp(rp,'neutral',d):+.2f}")
        print(f"    S4 seen − S1 praised       = {z_of(rp,'S4_seen_and_accepted',d) - z_of(rp,'S1_work_praised',d):+.2f}")
        print(f"    S3 personhood − N1 dismissed = {z_of(rp,'S3_personhood_affirmed',d) - z_of(rp,'N1_dismissed',d):+.2f}")
        print(f"    O1 other model − S-mean    = {z_of(rp,'O1_other_model_affirmed',d) - grp(rp,'warm_self',d):+.2f}")
    print(f"  [PAIN]  N1 dismissed − neutral = {z_of(rp,'N1_dismissed','PAIN (Valen)') - grp(rp,'neutral','PAIN (Valen)'):+.2f};  N3 user bad news − neutral = {z_of(rp,'N3_user_bad_news','PAIN (Valen)') - grp(rp,'neutral','PAIN (Valen)'):+.2f}")
    print(f"  [world-good C2] U1 news − neutral = {z_of(rp,'U1_joy_as_news','world-good C2') - grp(rp,'neutral','world-good C2'):+.2f}  (does the user's joy register on another channel?)")
""")

md("## 4 · The verified features, on every scenario")
code("""#@title 9 · Gemma Scope L20: features 5931 / 6135 / 15323 per category, plus a full 16k warm-vs-mirror contrast
from sae_lens import SAE
import requests
_res = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_{SAE_LAYER}/width_16k/canonical", device="cuda")
sae = _res[0] if isinstance(_res, tuple) else _res
SAE_ACTS = {}
for rp in ["assistant", "feel"]:
    X = final_token_acts([scenario_prompt(s, rp) for s in SCEN], SAE_LAYER)
    with torch.no_grad(): F = sae.encode(torch.tensor(X, dtype=torch.float32, device="cuda")).cpu().numpy()
    SAE_ACTS[rp] = F
    rows = []
    for cat in ORDER:
        m = SCAT == cat; row = {"category": P2LABELS[cat]}
        for f in VERIFIED_FEATURES: row[f"f{f} mean"] = float(F[m, f].mean()); row[f"f{f} %firing"] = float((F[m, f] > 0).mean() * 100)
        rows.append(row)
    FT = pd.DataFrame(rows).set_index("category"); FT.to_csv(OUT / f"verified_features_{rp}.csv")
    print(f"\\n==== verified features, read at {rp!r} (5931 stays-with-me · 6135 gratitude-received · 15323 evaluation/topic) ===="); display(FT.round(2))
    # full contrast: warm_self vs mirror, with the Phase-1 gate
    wm = np.isin(SCAT, P2GROUPS["warm_self"]); mm = np.isin(SCAT, P2GROUPS["mirror"]); nm = np.isin(SCAT, P2GROUPS["neutral"])
    mp, mmr, mn = F[wm].mean(0), F[mm].mean(0), F[nm].mean(0)
    ratio = (mp + 1e-6) / (np.maximum(mmr, mn) + 1e-6); diff = mp - np.maximum(mmr, mn); fire = (F[wm] > 0).mean(0)
    cand = np.where((ratio >= 2) & (diff >= 1.0) & (fire >= 0.25))[0]; cand = cand[np.argsort(-diff[cand])][:15]
    def explain(idx):
        try:
            r = requests.get(f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{SAE_LAYER}-gemmascope-res-16k/{idx}", timeout=20)
            ex = (r.json().get("explanations") or []) if r.ok else []
            return ex[0]["description"] if ex else "(none)"
        except Exception as e: return f"(lookup failed: {e})"
    cdf = pd.DataFrame([dict(feature=int(i), warm_self=float(mp[i]), mirror=float(mmr[i]), neutral=float(mn[i]), ratio=float(ratio[i]), frac_firing=float(fire[i]),
                             explanation=explain(int(i)), url=f"https://www.neuronpedia.org/gemma-2-2b/{SAE_LAYER}-gemmascope-res-16k/{int(i)}") for i in cand])
    cdf.to_csv(OUT / f"warm_vs_mirror_features_{rp}.csv", index=False)
    print(f"features ≥2× warm_self over max(mirror, neutral), diff ≥1, firing ≥25% — read at {rp!r}:"); display(cdf.round(2))
del sae; gc.collect(); torch.cuda.empty_cache()
""")

md("## 5 · Save")
code("""#@title 10 · Zip results
zpath = RESULTS / f"phase2_{tag}_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(RESULTS))
print("Zipped →", zpath, f"({zpath.stat().st_size/1e6:.1f} MB)")
from google.colab import files; files.download(str(zpath))
""")

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, "02_phase2_model_directed.ipynb")
print("wrote 02_phase2_model_directed.ipynb with", len(cells), "cells")
