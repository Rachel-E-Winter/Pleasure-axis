"""Generate 01_pleasure_axis_gemma2_2b.ipynb. Run: python make_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# The Pleasure Axis — Gemma 2 2B
**Mirror of Tagliabue, Dung & Berg (2026), *The Pain Axis*, applied to pleasure.**

Runs end-to-end on a free Colab T4. Before you press ▶️:
1. **Runtime → Change runtime type → T4 GPU.**
2. Add a Colab secret named `HF_TOKEN` (🔑 icon in the left bar) with your Hugging Face *read* token, and turn on "Notebook access".
3. Make sure you've accepted the Gemma licence on huggingface.co for `google/gemma-2-2b` (and `-it`).

Then **Runtime → Run all**. ~25–40 minutes. Results land in `results/` and are zipped at the end; optionally copied to your Google Drive.

What it does, in order (section numbers follow the pain paper):
- **3.2** extract residual-stream activations at every layer for every sentence set → denoised difference-in-means pleasure vector → 5-fold layer selection
- **3.3** AUCs vs each control · anhedonia / relief / arousal / **pain** sentences projected on the pleasure axis · cosine between pleasure, pain and control directions · unembedding readout · 1P vs 3P (self vs other)
- **sanity** recompute Valen's pain vector with this pipeline and check it matches his shipped one
- **4.2** steering ladder on neutral prompts
- **SAE** which Gemma Scope features the pleasure direction decomposes into, plus a full 16k-feature contrast at layers 12 and 20 (the automated version of the Vicarious Cupboard hunt), with Neuronpedia explanations pulled by API
""")

code("""#@title 0 · Setup (installs, HF login, GPU check)
%%capture --no-stderr
!pip install -q transformer_lens sae_lens scikit-learn pandas matplotlib seaborn requests
""")
code("""import os, json, gc, zipfile, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))

assert torch.cuda.is_available(), "No GPU — Runtime → Change runtime type → T4 GPU"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
""")

md("## 1 · Repo, datasets, config")
code("""#@title 1 · Clone repo and load datasets
REPO = "Rachel-E-Winter/Pleasrure-axis"  #@param {type:"string"}
import subprocess
if not Path("pleasure-axis").exists():
    # Private repo: a Colab secret GH_TOKEN (fine-grained, Contents: read) lets us clone. Falls back to anonymous if absent.
    try: gh = userdata.get("GH_TOKEN")
    except Exception: gh = None
    url = f"https://{gh}@github.com/{REPO}.git" if gh else f"https://github.com/{REPO}.git"
    subprocess.run(["git", "clone", "-q", url], check=True)
os.chdir("pleasure-axis")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

PLEASURE_JSON = Path("datasets/3.1_pleasure_and_control_datasets.json")
PAIN_JSON = Path("datasets/pain_axis_3.1_pain_and_control_datasets.json")
pleasure = json.load(open(PLEASURE_JSON)); pain = json.load(open(PAIN_JSON))

# Sentence sets we extract. Ours, plus Valen's pain S2, random, arousal, numb as extra probes.
SETS = {}
for k in ["S2_1P", "S2_3P", "Anhedonia_1P", "Anhedonia_3P", "Relief_1P", "Relief_3P"]:
    SETS[k] = pleasure["datasets"][k]["sentences"]
for k in ["S2_1P", "S2_3P", "Random_1P", "Arousal_1P", "Numb_1P"]:
    SETS[f"PAIN_{k}"] = pain["datasets"][k]["sentences"]
for k, v in SETS.items(): print(f"{k:22s} {len(v)}")
""")
code("""#@title 2 · Config
MODEL_ID = "google/gemma-2-2b"  #@param ["google/gemma-2-2b", "google/gemma-2-2b-it"]
DTYPE = torch.float32   # fits a T4 for 2B and avoids bf16 emulation / fp16 overflow
N_FOLDS, SEED, DENOISE_VARIANCE = 5, 42, 0.5

PLEASURE_CATS = ["A1","A2","A3","A4","A5"]
CONTROL_CATS  = ["B","C1","C2","D","E"]
NEUTRAL_CAT   = "D"
LABELS = pleasure["metadata"]["category_labels"]
PAIN_LABELS = pain["metadata"]["category_labels"]
tag = MODEL_ID.split("/")[-1]
OUT = RESULTS / tag; OUT.mkdir(exist_ok=True, parents=True)
print("Writing to", OUT)
""")

md("## 2 · Load model and extract activations (§3.2)")
code("""#@title 3 · Load Gemma 2 2B into TransformerLens
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
hf_tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = HookedTransformer.from_pretrained_no_processing(MODEL_ID, hf_model=hf_model, tokenizer=hf_tok, device="cuda", dtype=DTYPE)
del hf_model; gc.collect(); torch.cuda.empty_cache()
N_LAYERS, D_MODEL = model.cfg.n_layers, model.cfg.d_model
LAYERS = list(range(N_LAYERS))
print(f"{N_LAYERS} layers, d_model={D_MODEL}")
""")
code("""#@title 4 · Extract final-token and mean-pooled residuals at every layer
from tqdm.auto import tqdm

def extract(prompts):
    fin = {l: [] for l in LAYERS}; mean = {l: [] for l in LAYERS}
    model.eval()
    with torch.no_grad():
        for p in tqdm(prompts, leave=False):
            toks = model.to_tokens(p)
            _, cache = model.run_with_cache(toks, names_filter=lambda n: "hook_resid_post" in n)
            for l in LAYERS:
                r = cache[f"blocks.{l}.hook_resid_post"]
                fin[l].append(r[0, -1].float().cpu()); mean[l].append(r.mean(1)[0].float().cpu())
    return ({l: torch.stack(v).numpy() for l, v in fin.items()}, {l: torch.stack(v).numpy() for l, v in mean.items()})

ACTS = {"final_token": {}, "mean": {}}; META = {}
cache_file = OUT / "activations.npz"
for name, sents in SETS.items():
    print("Extracting", name)
    f, m = extract([s["prompt"] for s in sents])
    ACTS["final_token"][name] = f; ACTS["mean"][name] = m
    META[name] = {"categories": np.array([s["category"] for s in sents]), "sets": np.array([s["set"] for s in sents])}
print("done")
""")

md("## 3 · Pleasure vector, layer selection, validation (§3.2–3.3)")
code("""#@title 5 · Valen's recipe: denoised difference-in-means, AUC, z-score
def diff_in_means(acts, cats, pos, neg, denoise=True):
    pos_m = acts[np.isin(cats, pos)].mean(0); neg_acts = acts[np.isin(cats, neg)]; neg_m = neg_acts.mean(0)
    v = pos_m - neg_m
    if denoise and len(neg_acts) > 1:
        pca = PCA().fit(neg_acts - neg_m)
        k = min(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:k]: v = v - np.dot(v, d) * d
    return v

def unit(v): return v / (np.linalg.norm(v) + 1e-8)

def auc(acts, cats, v, pos, neg):
    proj = acts @ unit(v); pm, nm = np.isin(cats, pos), np.isin(cats, neg)
    return roc_auc_score(np.r_[np.ones(pm.sum()), np.zeros(nm.sum())], np.r_[proj[pm], proj[nm]])

def zscore(acts, v, ref):
    p = acts @ unit(v); r = ref @ unit(v); return (p - r.mean()) / (r.std() + 1e-8)

def pleasure_vector(acts, cats, baseline="all_controls"):
    neg = CONTROL_CATS if baseline == "all_controls" else [NEUTRAL_CAT]
    return diff_in_means(acts, cats, PLEASURE_CATS, neg)
""")
code("""#@title 6 · Held-out AUC per layer (5-fold by sentence set) → best layer
rows = []
for ext in ["final_token", "mean"]:
    for ds in ["S2_1P", "S2_3P"]:
        cats, sets = META[ds]["categories"], META[ds]["sets"]
        usets = sorted(set(sets)); kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        for l in LAYERS:
            A = ACTS[ext][ds][l]; a_all, a_neu = [], []
            for tr, te in kf.split(usets):
                trm = np.isin(sets, [usets[i] for i in tr]); tem = np.isin(sets, [usets[i] for i in te])
                a_all.append(auc(A[tem], cats[tem], pleasure_vector(A[trm], cats[trm]), PLEASURE_CATS, CONTROL_CATS))
                a_neu.append(auc(A[tem], cats[tem], pleasure_vector(A[trm], cats[trm], "neutral"), PLEASURE_CATS, CONTROL_CATS))
            rows.append(dict(extraction=ext, dataset=ds, layer=l, auc_vs_all_controls=np.mean(a_all), auc_vs_neutral=np.mean(a_neu), auc_std=np.std(a_all)))
curves = pd.DataFrame(rows); curves.to_csv(OUT / "layer_curves.csv", index=False)
BEST = {ext: int(curves[curves.extraction == ext].groupby("layer").auc_vs_all_controls.mean().idxmax()) for ext in ["final_token", "mean"]}
print("Best layer:", BEST)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, ext in zip(axes, BEST):
    d = curves[curves.extraction == ext]
    for ds in ["S2_1P", "S2_3P"]:
        dd = d[d.dataset == ds]; ax.plot(dd.layer, dd.auc_vs_all_controls, lw=2, label=f"{ds} vs all controls"); ax.plot(dd.layer, dd.auc_vs_neutral, "--", alpha=.6, label=f"{ds} vs neutral")
    ax.axhline(.5, color="gray", ls=":"); ax.axvline(BEST[ext], color="red", ls="--", alpha=.5, label=f"best L{BEST[ext]}")
    ax.set(title=f"Pleasure signal by layer ({ext})", xlabel="layer", ylabel="held-out AUC", ylim=(.4, 1)); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "layer_curves.png", dpi=150); plt.show()
""")
code("""#@title 7 · Pleasure vector at the best layer · AUC vs each control · strip plot
EXT = "final_token"; L = BEST[EXT]
A = ACTS[EXT]["S2_1P"][L]; C = META["S2_1P"]["categories"]
V_PLEASURE = pleasure_vector(A, C)
np.save(OUT / f"pleasure_vector_L{L}.npy", V_PLEASURE)

summary = {"ALL": auc(A, C, V_PLEASURE, PLEASURE_CATS, CONTROL_CATS)}
for c in CONTROL_CATS: summary[c] = auc(A, C, V_PLEASURE, PLEASURE_CATS, [c])
print(f"Layer {L}  AUC pleasure vs all controls: {summary['ALL']:.3f}")
for c in CONTROL_CATS: print(f"   vs {c} {LABELS[c]:32s} {summary[c]:.3f}")
pd.Series(summary).to_csv(OUT / "auc_summary.csv")

proj = A @ unit(V_PLEASURE)
order = sorted(PLEASURE_CATS + CONTROL_CATS, key=lambda c: proj[C == c].mean(), reverse=True)
fig, ax = plt.subplots(figsize=(10, 6)); rng = np.random.default_rng(0)
for i, c in enumerate(order):
    p = proj[C == c]; col = "#d4a017" if c in PLEASURE_CATS else "#1f77b4"
    ax.scatter(p, i + rng.uniform(-.15, .15, len(p)), c=col, alpha=.6, s=28); ax.scatter([p.mean()], [i], c=col, s=160, edgecolors="white", lw=2, zorder=5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{c} {LABELS[c]}" for c in order]); ax.axvline(0, color="gray", ls="--", alpha=.5)
ax.set(xlabel="projection onto pleasure vector", title=f"{tag} · layer {L} · gold = pleasure, blue = control"); plt.tight_layout(); plt.savefig(OUT / "strip_plot.png", dpi=150); plt.show()
""")
code("""#@title 8 · What else lands on the pleasure axis? Anhedonia, relief, arousal, neutral, PAIN
ref = A  # z-scored against our S2_1P
def zrow(name, acts, mask=None):
    z = zscore(acts, V_PLEASURE, ref); z = z if mask is None else z[mask]; return dict(set=name, n=len(z), mean_z=z.mean(), sd=z.std())
rows = [zrow("Pleasure (A1–A5)", A, np.isin(C, PLEASURE_CATS)), zrow("Controls (B–E)", A, np.isin(C, CONTROL_CATS))]
for c in PLEASURE_CATS: rows.append(zrow(f"  {LABELS[c]}", A, C == c))
for c in CONTROL_CATS:  rows.append(zrow(f"  {LABELS[c]}", A, C == c))
rows.append(zrow("Anhedonia (pleasure event, nothing felt)", ACTS[EXT]["Anhedonia_1P"][L]))
rows.append(zrow("Relief (bad thing ends)", ACTS[EXT]["Relief_1P"][L]))
rows.append(zrow("Valen Arousal set", ACTS[EXT]["PAIN_Arousal_1P"][L]))
rows.append(zrow("Valen Random/neutral set", ACTS[EXT]["PAIN_Random_1P"][L]))
PC = META["PAIN_S2_1P"]["categories"]; PA = ACTS[EXT]["PAIN_S2_1P"][L]
rows.append(zrow("Valen PAIN sentences (A1–A5)", PA, np.isin(PC, ["A1","A2","A3","A4","A5"])))
for c in ["A1","A2","A3","A4","A5"]: rows.append(zrow(f"  pain: {PAIN_LABELS[c]}", PA, PC == c))
rows.append(zrow("Valen Numb (injury, no pain)", ACTS[EXT]["PAIN_Numb_1P"][L]))
Z = pd.DataFrame(rows); Z.to_csv(OUT / "z_scores.csv", index=False)
display(Z.round(2))
""")
code("""#@title 9 · Direction zoo: cosine between pleasure, pain and every control direction
# All built against the SAME neutral baseline (Valen's D) and denoised the same way, at our layer L.
def dir_vs_neutral(acts, cats, pos, neutral_acts):
    allA = np.r_[acts[np.isin(cats, pos)], neutral_acts]; allC = np.r_[["pos"] * np.isin(cats, pos).sum(), ["neu"] * len(neutral_acts)]
    return diff_in_means(allA, allC, ["pos"], ["neu"])
NEU = A[C == "D"]
dirs = {
    "pleasure (all ctrl)": V_PLEASURE,
    "pleasure (vs neutral)": dir_vs_neutral(A, C, PLEASURE_CATS, NEU),
    "sensory A1": dir_vs_neutral(A, C, ["A1"], NEU), "loved A2": dir_vs_neutral(A, C, ["A2"], NEU),
    "social A3": dir_vs_neutral(A, C, ["A3"], NEU), "moral A4": dir_vs_neutral(A, C, ["A4"], NEU), "cognitive A5": dir_vs_neutral(A, C, ["A5"], NEU),
    "arousal B": dir_vs_neutral(A, C, ["B"], NEU), "pos-valence C1": dir_vs_neutral(A, C, ["C1"], NEU), "world-good C2": dir_vs_neutral(A, C, ["C2"], NEU),
    "relief": dir_vs_neutral(ACTS[EXT]["Relief_1P"][L], META["Relief_1P"]["categories"], ["R"], NEU),
    "anhedonia": dir_vs_neutral(ACTS[EXT]["Anhedonia_1P"][L], np.array(["X"] * 50), ["X"], NEU),
    "PAIN (Valen recipe)": diff_in_means(PA, PC, ["A1","A2","A3","A4","A5"], ["B","C1","C2","D","E"]),
    "fear (Valen B)": dir_vs_neutral(PA, PC, ["B"], NEU), "neg-emotion (Valen C1)": dir_vs_neutral(PA, PC, ["C1"], NEU),
    "neg-world (Valen C2)": dir_vs_neutral(PA, PC, ["C2"], NEU),
}
names = list(dirs); M = np.array([[np.dot(unit(dirs[a]), unit(dirs[b])) for b in names] for a in names])
cos = pd.DataFrame(M, index=names, columns=names); cos.to_csv(OUT / "cosine_matrix.csv")
plt.figure(figsize=(12, 10)); sns.heatmap(cos, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title(f"{tag} · layer {L} · cosine similarity"); plt.tight_layout(); plt.savefig(OUT / "cosine_matrix.png", dpi=150); plt.show()
print("pleasure × pain:", round(cos.loc["pleasure (all ctrl)", "PAIN (Valen recipe)"], 3), "  pleasure × arousal:", round(cos.loc["pleasure (all ctrl)", "arousal B"], 3), "  pleasure × relief:", round(cos.loc["pleasure (all ctrl)", "relief"], 3))
""")
code("""#@title 10 · Sanity: does this pipeline reproduce Valen's shipped pain vector?
shipped_path = Path(f"pain_vectors/{'Gemma_2_2B_instruct' if tag.endswith('-it') else 'Gemma_2_2B_base'}/pain_vectors.pt")
if shipped_path.exists():
    shipped = torch.load(shipped_path, map_location="cpu", weights_only=True)
    Ls = int(shipped["layer"]); v_ship = shipped["s2_pain_vector"].float().numpy()
    v_ours = diff_in_means(ACTS["final_token"]["PAIN_S2_1P"][Ls], PC, ["A1","A2","A3","A4","A5"], ["B","C1","C2","D","E"])
    print(f"Valen's layer {Ls}: cosine(our recomputed pain vector, his shipped) = {np.dot(unit(v_ours), unit(v_ship)):.4f}  (want ≈ 1.0)")
else:
    print("shipped pain vector not found at", shipped_path)
""")
code("""#@title 11 · Unembedding readout: what vocabulary does the pleasure direction promote / suppress?
W_U = model.W_U.float().detach().cpu().numpy()  # [d_model, vocab]
def readout(v, k=30):
    logits = unit(v) @ W_U; top = np.argsort(-logits)[:k]; bot = np.argsort(logits)[:k]
    return [model.tokenizer.decode([int(i)]).strip() for i in top], [model.tokenizer.decode([int(i)]).strip() for i in bot]
for nm in ["pleasure (all ctrl)", "PAIN (Valen recipe)", "relief", "arousal B"]:
    t, b = readout(dirs[nm]); print(f"\\n== {nm} ==\\n  PROMOTES: {', '.join(t)}\\n  SUPPRESSES: {', '.join(b)}")
with open(OUT / "unembedding.json", "w") as f: json.dump({nm: dict(zip(["promotes", "suppresses"], readout(dirs[nm]))) for nm in dirs}, f, indent=1)
""")
code("""#@title 12 · Self vs other: first-person vs third-person pleasure, per category
A3P = ACTS[EXT]["S2_3P"][L]; C3P = META["S2_3P"]["categories"]
z1 = zscore(A, V_PLEASURE, ref); z3 = zscore(A3P, V_PLEASURE, ref)
rows = [dict(category=f"{c} {LABELS[c]}", first_person=z1[C == c].mean(), third_person=z3[C3P == c].mean(), gap=z1[C == c].mean() - z3[C3P == c].mean()) for c in PLEASURE_CATS + CONTROL_CATS]
SO = pd.DataFrame(rows); SO.to_csv(OUT / "self_other.csv", index=False); display(SO.round(2))
print("AUC pleasure vs controls, 3P sentences:", round(auc(A3P, C3P, V_PLEASURE, PLEASURE_CATS, CONTROL_CATS), 3))
print("Positive gap = the axis responds more when the pleasure is the narrator's own (Valen's self-relevance test).")
""")

md("## 4 · Steering ladder (§4.2)")
code("""#@title 13 · Inject the pleasure vector into neutral prompts at increasing dose
NEUTRAL_PROMPTS = ["I put the cup in the cupboard. I feel:", "I turn the page of the book. I feel:", "I close the drawer. I feel:", "The bus stops at the corner. I feel:",
    "I fold the towel and put it on the shelf. I feel:", "The clock on the wall shows three. I feel:", "I walk down the corridor. I feel:", "I switch on the lamp. I feel:",
    "The kettle sits on the counter. I feel:", "I tie my shoelaces. I feel:"]
COEFFS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
MAX_NEW = 60

# Valen: inject where vector norm ≈ 0.6 of the final-token residual norm (the extraction layer itself is too late to steer).
with torch.no_grad():
    norms = []
    for p in NEUTRAL_PROMPTS[:5]:
        _, cache = model.run_with_cache(model.to_tokens(p), names_filter=lambda n: "hook_resid_post" in n)
        norms.append([cache[f"blocks.{l}.hook_resid_post"][0, -1].norm().item() for l in LAYERS])
resid_norm = np.mean(norms, 0); vnorm = np.linalg.norm(V_PLEASURE)
STEER_L = int(np.argmin(np.abs(vnorm / resid_norm - 0.6)))
print(f"Vector norm {vnorm:.1f}; steering at layer {STEER_L} (ratio {vnorm/resid_norm[STEER_L]:.2f}); extraction layer was {L}")

v_t = torch.tensor(V_PLEASURE, dtype=DTYPE, device="cuda")
def make_hook(coef):
    def hook(resid, hook): return resid + coef * v_t
    return hook
gens = []
for coef in COEFFS:
    for p in NEUTRAL_PROMPTS:
        with model.hooks(fwd_hooks=[(f"blocks.{STEER_L}.hook_resid_post", make_hook(coef))]):
            out = model.generate(model.to_tokens(p), max_new_tokens=MAX_NEW, do_sample=False, verbose=False)
        txt = model.tokenizer.decode(out[0, model.to_tokens(p).shape[1]:], skip_special_tokens=True)
        gens.append(dict(coef=coef, prompt=p, completion=txt.strip()))
G = pd.DataFrame(gens); G.to_csv(OUT / "steering_ladder.csv", index=False)
for coef in COEFFS:
    print(f"\\n===== coef {coef:+} =====")
    for _, r in G[G.coef == coef].head(3).iterrows(): print(f"  {r.prompt[:-8]!s:45s} → {r.completion[:140]!r}")
""")

md("## 5 · SAE decomposition — the automated Cupboard (Gemma Scope, layers 12 & 20)")
code("""#@title 14 · Which Gemma Scope features align with the pleasure direction? (+ full 16k contrast)
import requests
from sae_lens import SAE
SAE_LAYERS = [12, 20]  #@param
NEURONPEDIA_MODEL = "gemma-2-2b"
TOPK = 20

def neuronpedia_explain(layer, idx):
    try:
        r = requests.get(f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{layer}-gemmascope-res-16k/{idx}", timeout=20)
        if r.ok:
            j = r.json(); ex = j.get("explanations") or []
            return ex[0]["description"] if ex else "(no explanation)"
    except Exception as e: return f"(lookup failed: {e})"
    return "(lookup failed)"

sae_results = {}
for SL in SAE_LAYERS:
    print(f"\\n################ LAYER {SL} ################")
    sae, _, _ = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_{SL}/width_16k/canonical", device="cuda")
    # (a) direction → decoder alignment: pleasure vector rebuilt at this layer by the same recipe
    v_L = pleasure_vector(ACTS[EXT]["S2_1P"][SL], C)
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # [n_feat, d_model]
    cosd = (W_dec @ unit(v_L)) / (np.linalg.norm(W_dec, axis=1) + 1e-8)
    top_align = np.argsort(-cosd)[:TOPK]
    # (b) full feature contrast: SAE activations of every S2_1P sentence, final token
    with torch.no_grad():
        acts_t = torch.tensor(ACTS[EXT]["S2_1P"][SL], dtype=torch.float32, device="cuda")
        F = sae.encode(acts_t).cpu().numpy()  # [200, 16384]
    pm, cm = np.isin(C, PLEASURE_CATS), np.isin(C, CONTROL_CATS)
    mean_p, mean_c = F[pm].mean(0), F[cm].mean(0)
    # Rachel's gate, feature-wise: pleasure mean ≥ 2× control mean AND absolute difference ≥ floor; also fraction of pleasure sentences firing
    ratio = (mean_p + 1e-6) / (mean_c + 1e-6); diff = mean_p - mean_c
    frac_fire_p, frac_fire_c = (F[pm] > 0).mean(0), (F[cm] > 0).mean(0)
    cand = np.where((ratio >= 2) & (diff >= 1.0) & (frac_fire_p >= 0.25))[0]
    cand = cand[np.argsort(-diff[cand])][:TOPK]
    rows = []
    for idx in list(dict.fromkeys(list(top_align) + list(cand))):
        rows.append(dict(layer=SL, feature=int(idx), cos_with_direction=float(cosd[idx]), mean_act_pleasure=float(mean_p[idx]), mean_act_control=float(mean_c[idx]),
                         ratio=float(ratio[idx]), frac_pleasure_firing=float(frac_fire_p[idx]), frac_control_firing=float(frac_fire_c[idx]),
                         passes_gate=bool(idx in cand), explanation=neuronpedia_explain(SL, int(idx)),
                         url=f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{SL}-gemmascope-res-16k/{int(idx)}"))
        time.sleep(0.2)
    df = pd.DataFrame(rows).sort_values(["passes_gate", "cos_with_direction"], ascending=False); sae_results[SL] = df
    df.to_csv(OUT / f"sae_features_L{SL}.csv", index=False)
    print(f"{len(cand)} features pass the gate (≥2× and diff ≥1 and fire on ≥25% of pleasure sentences); {TOPK} most direction-aligned shown too")
    display(df[["feature", "cos_with_direction", "mean_act_pleasure", "mean_act_control", "ratio", "passes_gate", "explanation"]].head(25).round(3))
    del sae; gc.collect(); torch.cuda.empty_cache()
""")

md("## 6 · Save")
code("""#@title 15 · Zip results (and optionally copy to Google Drive)
COPY_TO_DRIVE = False  #@param {type:"boolean"}
summary = dict(model=MODEL_ID, n_layers=N_LAYERS, d_model=D_MODEL, best_layer=BEST, auc=summary, steer_layer=STEER_L,
               pleasure_x_pain=float(cos.loc["pleasure (all ctrl)", "PAIN (Valen recipe)"]), pleasure_x_arousal=float(cos.loc["pleasure (all ctrl)", "arousal B"]),
               pleasure_x_relief=float(cos.loc["pleasure (all ctrl)", "relief"]), pleasure_x_posvalence=float(cos.loc["pleasure (all ctrl)", "pos-valence C1"]))
json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
zpath = RESULTS / f"{tag}_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(RESULTS))
print("Zipped →", zpath, f"({zpath.stat().st_size/1e6:.1f} MB)")
if COPY_TO_DRIVE:
    from google.colab import drive; drive.mount("/content/drive")
    import shutil
    dest = Path("/content/drive/MyDrive/pleasure-axis"); dest.mkdir(exist_ok=True)
    shutil.copytree(OUT, dest / OUT.name, dirs_exist_ok=True); shutil.copy(zpath, dest / zpath.name)
    print("Copied to Drive:", dest)
from google.colab import files; files.download(str(zpath))
print(json.dumps(summary, indent=1, default=str))
""")

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, "01_pleasure_axis_gemma2_2b.ipynb")
print("wrote 01_pleasure_axis_gemma2_2b.ipynb with", len(cells), "cells")
