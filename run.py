#!/usr/bin/env python3
"""
run.py — Contrastive representation approach for hallucination detection.

Idea: Early transformer layers (L3) encode factual knowledge BEFORE reasoning
layers distort it. By contrasting representations of hallucinated vs correct
answers, we find a linear subspace that separates them.

Architecture:
  1. Teacher-forcing: feed question+answer, capture hidden states at L3, L15
  2. Compute uncertainty metrics from output logits
  3. PCA-compress hidden states + project onto contrastive direction
  4. LogisticRegression classifier

Best public PR-AUC: ~0.844
"""
import torch, torch.nn as nn, numpy as np, pandas as pd
import os, sys, json, warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# Model patches (GigaChat uses DeepseekV3 arch with quirks)
# ════════════════════════════════════════════════════════════
def patch_gigachat():
    import transformers.models.deepseek_v3.configuration_deepseek_v3 as cfm
    orig_init = cfm.DeepseekV3Config.__init__
    def new_init(self, *a, **kw):
        if 'routed_scaling_factor' in kw:
            kw['routed_scaling_factor'] = float(kw['routed_scaling_factor'])
        rs = kw.get('rope_scaling')
        if rs:
            for k in ['beta_fast', 'beta_slow', 'factor']:
                if k in rs and isinstance(rs[k], int): rs[k] = float(rs[k])
        orig_init(self, *a, **kw)
    cfm.DeepseekV3Config.__init__ = new_init

    import transformers.models.deepseek_v3.modeling_deepseek_v3 as mdl
    orig_ai = mdl.DeepseekV3Attention.__init__
    def new_ai(self, config, layer_idx):
        sv = config.q_lora_rank
        if sv is None: config.q_lora_rank = config.hidden_size
        try: orig_ai(self, config, layer_idx)
        finally: config.q_lora_rank = sv
        if sv is None:
            del self.q_a_proj; del self.q_a_layernorm; del self.q_b_proj
            qhd = config.qk_nope_head_dim + config.qk_rope_head_dim
            self.q_proj = nn.Linear(config.hidden_size,
                                     config.num_attention_heads * qhd,
                                     bias=config.attention_bias)
            self.q_lora_rank = None
    mdl.DeepseekV3Attention.__init__ = new_ai

    orig_af = mdl.DeepseekV3Attention.forward
    def new_af(self, hs, *a, **kw):
        if getattr(self, 'q_lora_rank', -1) is None:
            object.__setattr__(self, 'q_a_proj', lambda x: x)
            object.__setattr__(self, 'q_a_layernorm', lambda x: x)
            object.__setattr__(self, 'q_b_proj', self.q_proj)
            try: return orig_af(self, hs, *a, **kw)
            finally:
                for nm in ['q_a_proj', 'q_a_layernorm', 'q_b_proj']:
                    object.__delattr__(self, nm)
        return orig_af(self, hs, *a, **kw)
    mdl.DeepseekV3Attention.forward = new_af

patch_gigachat()
from transformers import AutoModelForCausalLM, AutoTokenizer


# ════════════════════════════════════════════════════════════
# Uncertainty metrics
# ════════════════════════════════════════════════════════════
def logit_metrics(logits, ids, start):
    """Extract 18 uncertainty features from logits over answer span."""
    L = ids.shape[1]
    na = L - start
    if na <= 1: return None

    lgt = logits[0, start-1:L-1].float()
    toks = ids[0, start:L]
    p = torch.softmax(lgt, -1)
    lp = torch.log_softmax(lgt, -1)
    tok_lp = lp.gather(1, toks.unsqueeze(1)).squeeze(-1)
    h = -(p * torch.log(p + 1e-10)).sum(-1)
    top1 = p.max(-1).values
    top5 = p.topk(min(5, p.shape[-1]), -1).values.sum(-1)
    t2 = p.topk(min(2, p.shape[-1]), -1).values
    gap = t2[:, 0] - (t2[:, 1] if t2.shape[1] > 1 else torch.zeros_like(t2[:, 0]))
    m = max(1, na // 2)
    half1 = tok_lp[:m].mean().item()
    half2 = tok_lp[m:].mean().item() if na > 1 else half1

    return np.array([
        tok_lp.mean().item(), tok_lp.min().item(), tok_lp.max().item(),
        tok_lp.std().item() if na > 1 else 0.,
        tok_lp[0].item(), half1 - half2,
        h.mean().item(), h.max().item(), h.min().item(),
        h.std().item() if na > 1 else 0.,
        top1.mean().item(), top1.min().item(), top5.mean().item(),
        gap.mean().item(), gap.min().item(),
        torch.exp(-tok_lp.mean()).item(), float(na),
        (tok_lp < -5.).float().mean().item(),
    ], np.float32)


# ════════════════════════════════════════════════════════════
# Feature pipeline
# ════════════════════════════════════════════════════════════
class ContrastiveProbe:
    """Builds features from cached layer activations."""

    def __init__(self, cfg):
        self.layers = cfg['probe_layers']
        self.ns = cfg['n_scalars']
        self.pc_first = cfg['pca_components']['first']
        self.pc_mean = cfg['pca_components']['mean']
        self.C = cfg['regularization']
        self.cw = cfg.get('class_weight')
        self.gw = cfg.get('gen_oversample', 1)
        self.dir_src = cfg.get('direction_source', {})

    def fit(self, feat_dir='features', data_dir='data'):
        """Train PCA + directions + classifier from cached features."""
        tr = dict(np.load(f'{feat_dir}/train_all_layers.npz', allow_pickle=True))
        gn = dict(np.load(f'{feat_dir}/gen_all_layers.npz', allow_pickle=True))

        # --- labels ---
        parts = []
        for fn in ['training_data_labeled.csv', 'training_data_labeled_hot.csv',
                    'training_data_labeled_subtle.csv']:
            fp = f'{data_dir}/training/{fn}'
            if os.path.exists(fp):
                d = pd.read_csv(fp)
                d = d[d['is_hallucination'].notna()].reset_index(drop=True)
                if 'model_answer' in d.columns:
                    d = d[~d['model_answer'].astype(str).str.startswith('[ERROR')].reset_index(drop=True)
                parts.append(d)
        df_tr = pd.concat(parts, ignore_index=True)
        yt = df_tr['is_hallucination'].values
        ti = tr['indices']
        y_old = np.array([yt[i] for i in ti if i < len(yt)])
        nt = len(y_old)

        for lp in [f'{data_dir}/generated/bench_verified_labels_all.csv',
                    f'{data_dir}/generated/bench_verified_labels.csv']:
            if os.path.exists(lp):
                gl = pd.read_csv(lp).sort_values('index').reset_index(drop=True); break
        ng = len(gn['indices'])
        y_gen = gl['is_hallucination'].values[:ng]

        print(f'  Old: {nt} ({y_old.mean():.1%}), Gen: {ng} ({y_gen.mean():.1%})')

        # --- build matrix ---
        sc_tr = np.vstack([tr['scalars'][:nt, :self.ns]] +
                           [gn['scalars'][:ng, :self.ns]] * self.gw)
        yc = np.concatenate([y_old] + [y_gen] * self.gw)

        cols = [sc_tr]
        self.pcas = {}
        self.dirs = {}

        for li in self.layers:
            for sfx in ['', '_mean']:
                k = f'L{li}{sfx}'
                dim = self.pc_mean if sfx else self.pc_first
                mat = np.vstack([tr[k][:nt]] + [gn[k][:ng]] * self.gw)

                nc = min(dim, mat.shape[0]-1, mat.shape[1])
                pca = PCA(n_components=nc, random_state=0)
                cols.append(pca.fit_transform(mat))
                self.pcas[k] = pca

                src = self.dir_src.get(k, 'gen')
                if src == 'train':
                    dp, dy = tr[k][:nt], y_old
                else:
                    dp, dy = gn[k][:ng], y_gen
                d = dp[dy == 1].mean(0) - dp[dy == 0].mean(0)
                d /= (np.linalg.norm(d) + 1e-10)
                self.dirs[k] = d
                cols.append((mat @ d).reshape(-1, 1))

        X = np.hstack(cols)
        print(f'  Dim: {X.shape[1]}')

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.clf = LogisticRegression(C=self.C, max_iter=5000,
                                       class_weight=self.cw, random_state=0)
        self.clf.fit(Xs, yc)

        # --- public eval ---
        te = dict(np.load(f'{feat_dir}/test_all_layers.npz', allow_pickle=True))
        pub = pd.read_csv(f'{data_dir}/bench/knowledge_bench_public.csv')
        yi = te['indices']
        yp = pub['is_hallucination'].values[yi]
        nte = len(yp)

        cols_te = [te['scalars'][:nte, :self.ns]]
        for li in self.layers:
            for sfx in ['', '_mean']:
                k = f'L{li}{sfx}'
                ep = te[k][:nte]
                cols_te.append(self.pcas[k].transform(ep))
                cols_te.append((ep @ self.dirs[k]).reshape(-1, 1))

        Xte = np.hstack(cols_te)
        pred = self.clf.predict_proba(self.scaler.transform(Xte))[:, 1]
        score = average_precision_score(yp, pred)
        print(f'  Public PR-AUC: {score:.4f}')
        return score

    def predict_single(self, feats):
        """Predict from raw extracted features dict."""
        cols = [feats['scalars'][:self.ns].reshape(1, -1)]
        for li in self.layers:
            for sfx in ['', '_mean']:
                k = f'L{li}{sfx}'
                v = feats[k].reshape(1, -1)
                cols.append(self.pcas[k].transform(v))
                cols.append((v @ self.dirs[k]).reshape(1, 1))
        x = np.hstack(cols)
        return self.clf.predict_proba(self.scaler.transform(x))[0, 1]


# ════════════════════════════════════════════════════════════
# Extraction
# ════════════════════════════════════════════════════════════
def attach_hooks(model, layers):
    store = {}; handles = []
    for i in layers:
        def mk(idx):
            def fn(mod, inp, out):
                store[idx] = (out[0] if isinstance(out, tuple) else out).detach()
            return fn
        handles.append(model.model.layers[i].register_forward_hook(mk(i)))
    return store, handles


def get_features(model, tok, dev, store, question, answer, layers):
    """Teacher-force and extract probe + uncertainty features."""
    m1 = [{"role": "user", "content": question}]
    m2 = m1 + [{"role": "assistant", "content": answer}]

    e1 = tok.apply_chat_template(m1, add_generation_prompt=True, return_tensors="pt")
    p_ids = (e1["input_ids"] if isinstance(e1, dict) else e1)
    plen = p_ids.shape[1]

    e2 = tok.apply_chat_template(m2, return_tensors="pt")
    f_ids = (e2["input_ids"] if isinstance(e2, dict) else e2).to(dev)
    seq = f_ids.shape[1]

    if seq - plen <= 1:
        return None

    with torch.no_grad():
        out = model(f_ids).logits

    sc = logit_metrics(out, f_ids, plen)
    if sc is None:
        del out; return None

    res = {'scalars': sc}
    for li in layers:
        if li in store:
            h = store[li][0]
            res[f'L{li}'] = h[plen - 1].cpu().float().numpy()
            res[f'L{li}_mean'] = h[plen:seq].mean(0).cpu().float().numpy()
    del out
    return res


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    cfg = json.load(open('configs/config.json'))
    layers = cfg['probe_layers']

    # Find private test
    priv = None
    for c in ['data/bench/knowledge_bench_private.csv',
              'data/bench/knowledge_bench_private_no_labels.csv'] + \
             ([sys.argv[1]] if len(sys.argv) > 1 else []):
        if c and os.path.exists(c):
            priv = c; break
    if not priv:
        print("No private test found"); sys.exit(1)

    df = pd.read_csv(priv)
    qc = next((c for c in ['prompt', 'question'] if c in df.columns), None)
    ac = next((c for c in ['model_answer', 'response'] if c in df.columns), None)
    has_ans = ac and df[ac].notna().sum() > 0
    print(f'Input: {priv} ({len(df)} rows, answers={has_ans})')

    # Train
    probe = ContrastiveProbe(cfg)
    probe.fit()

    # Load LLM
    print('Loading model...')
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg['model_id'], torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    dev = next(mdl.parameters()).device
    store, handles = attach_hooks(mdl, layers)

    # Score
    preds = []
    errs = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Scoring'):
        q = str(row[qc])
        if has_ans and pd.notna(row[ac]) and str(row[ac]).strip():
            a = str(row[ac])
        else:
            ms = [{"role": "user", "content": q}]
            enc = tok.apply_chat_template(ms, add_generation_prompt=True, return_tensors="pt")
            pid = (enc["input_ids"] if isinstance(enc, dict) else enc).to(dev)
            with torch.no_grad():
                o = mdl.generate(pid, max_new_tokens=150, temperature=0.01,
                                  do_sample=True, repetition_penalty=1.1)
            a = tok.decode(o[0, pid.shape[1]:], skip_special_tokens=True).strip()

        try:
            ft = get_features(mdl, tok, dev, store, q, a, layers)
            pr = probe.predict_single(ft) if ft else 0.5
            if ft is None: errs += 1
        except Exception as e:
            print(f'  err {i}: {e}'); pr = 0.5; errs += 1

        preds.append(pr)
        if i % 50 == 0: torch.cuda.empty_cache()

    for h in handles: h.remove()

    out = pd.DataFrame({'predict_proba': preds})
    op = 'data/bench/knowledge_bench_private_scores.csv'
    out.to_csv(op, index=False)
    print(f'\n✅ {op}: {len(preds)} predictions, mean={np.mean(preds):.3f}, errors={errs}')


if __name__ == '__main__':
    main()
