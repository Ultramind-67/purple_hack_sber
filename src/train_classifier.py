import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. ПАТЧИ ДЛЯ DEEPSEEK-V3 (GigaChat)
# ==========================================
import transformers.models.deepseek_v3.configuration_deepseek_v3 as cfg_mod
_orig_cfg_init = cfg_mod.DeepseekV3Config.__init__
def _patched_cfg_init(self, *args, **kwargs):
    if 'routed_scaling_factor' in kwargs: kwargs['routed_scaling_factor'] = float(kwargs['routed_scaling_factor'])
    rs = kwargs.get('rope_scaling')
    if rs:
        for k in ['beta_fast', 'beta_slow', 'factor']:
            if k in rs and isinstance(rs[k], int): rs[k] = float(rs[k])
    _orig_cfg_init(self, *args, **kwargs)
cfg_mod.DeepseekV3Config.__init__ = _patched_cfg_init

import transformers.models.deepseek_v3.modeling_deepseek_v3 as dsv3
_orig_attn_init = dsv3.DeepseekV3Attention.__init__
def _patched_attn_init(self, config, layer_idx):
    saved = config.q_lora_rank
    if saved is None: config.q_lora_rank = config.hidden_size
    try: _orig_attn_init(self, config, layer_idx)
    finally: config.q_lora_rank = saved
    if saved is None:
        del self.q_a_proj; del self.q_a_layernorm; del self.q_b_proj
        qhd = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * qhd, bias=config.attention_bias)
        self.q_lora_rank = None
dsv3.DeepseekV3Attention.__init__ = _patched_attn_init

_orig_attn_fwd = dsv3.DeepseekV3Attention.forward
def _patched_attn_fwd(self, hidden_states, *a, **kw):
    if getattr(self, 'q_lora_rank', -1) is None:
        object.__setattr__(self, 'q_a_proj', lambda x: x)
        object.__setattr__(self, 'q_a_layernorm', lambda x: x)
        object.__setattr__(self, 'q_b_proj', self.q_proj)
        try: return _orig_attn_fwd(self, hidden_states, *a, **kw)
        finally:
            object.__delattr__(self, 'q_a_proj'); object.__delattr__(self, 'q_a_layernorm'); object.__delattr__(self, 'q_b_proj')
    return _orig_attn_fwd(self, hidden_states, *a, **kw)
dsv3.DeepseekV3Attention.forward = _patched_attn_fwd

# ==========================================
# 2. FEATURE ACCUMULATOR
# ==========================================
class GuardianAccumulator:
    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.model.layers)
        self.probe_layers = [0, 5, 10, 15, 20, 25]
        self._hooks = []
        self._hidden = {}
        self._router = {}

    def attach(self):
        self._hidden.clear()
        self._router.clear()
        for idx in range(self.num_layers):
            layer = self.model.model.layers[idx]
            if idx in self.probe_layers:
                def make_hs_hook(i):
                    def hook(mod, inp, out):
                        self._hidden[i] = (out[0] if isinstance(out, tuple) else out).detach()
                    return hook
                self._hooks.append(layer.register_forward_hook(make_hs_hook(idx)))
            
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                def make_router_hook(i):
                    def hook(mod, inp, out):
                        self._router[i] = out[1].detach() # out[1] contains weights of top-k experts
                    return hook
                self._hooks.append(layer.mlp.gate.register_forward_hook(make_router_hook(idx)))

    def detach(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def __enter__(self): self.attach(); return self
    def __exit__(self, *_): self.detach()

    def compute_features(self, logits, input_ids, answer_start):
        seq_len = input_ids.shape[1]
        n_answer = seq_len - answer_start
        if n_answer <= 0: return None
        
        # 1. Uncertainty Features
        ans_logits = logits[0, answer_start-1 : seq_len-1].float()
        ans_ids = input_ids[0, answer_start : seq_len]
        probs = torch.softmax(ans_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        log_probs = torch.log_softmax(ans_logits, dim=-1)
        token_lp = log_probs.gather(1, ans_ids.unsqueeze(1)).squeeze(-1)
        
        top1 = probs.max(dim=-1).values
        top5 = probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1)
        top2 = probs.topk(min(2, probs.shape[-1]), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0)
        
        unc_feats = [
            token_lp.mean().item(), token_lp.min().item(), token_lp.max().item(),
            token_lp.std().item() if n_answer > 1 else 0.0,
            entropy.mean().item(), entropy.max().item(), entropy.std().item() if n_answer > 1 else 0.0,
            margin.mean().item(), margin.min().item(),
            top1.mean().item(), top5.mean().item(),
            float(n_answer)
        ]
        
        # 2. Internal Scalars & Logit Lens
        int_feats = []
        last_hs = self._hidden[self.probe_layers[-1]][0]
        probe_vec = last_hs[answer_start - 1].cpu().float().numpy()
        
        for idx in self.probe_layers:
            hs = self._hidden[idx][0]
            int_feats.append(hs[answer_start - 1].norm().item())
            if idx in [0, 15, 25]:
                ans_hs = hs[answer_start - 1 : seq_len - 1].unsqueeze(0)
                with torch.no_grad():
                    ll = self.model.lm_head(self.model.model.norm(ans_hs)).float()
                ll_p = torch.softmax(ll[0], dim=-1)
                ll_e = -(ll_p * torch.log(ll_p + 1e-10)).sum(dim=-1)
                int_feats.append(ll_e.mean().item())

        # 3. MoE Routing Features
        router_feats = []
        if self._router:
            confs = []
            for l_idx, r_weights in self._router.items():
                if r_weights.shape[0] > answer_start:
                    confs.append(r_weights[answer_start:, 0].mean().item())
            if confs: router_feats.extend([np.mean(confs), np.std(confs), np.min(confs)])
            else: router_feats.extend([0.0, 0.0, 0.0])
        else:
            router_feats.extend([0.0, 0.0, 0.0])

        self._hidden.clear()
        self._router.clear()
        
        scalars = np.array(unc_feats + int_feats + router_feats, dtype=np.float32)
        return {"scalars": scalars, "probe_vec": probe_vec}

# ==========================================
# 3. UTILS
# ==========================================
def extract_dataset_features(df, model, tokenizer, accumulator, device):
    X_scalars, X_probes, Y = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        prompt, answer = row['prompt'], row['model_answer']
        if pd.isna(answer): continue
        messages_prompt = [{"role": "user", "content": prompt}]
        messages_full = messages_prompt + [{"role": "assistant", "content": answer}]
        prompt_enc = tokenizer.apply_chat_template(messages_prompt, add_generation_prompt=True, tokenize=True)
        full_enc = tokenizer.apply_chat_template(messages_full, tokenize=True)
        answer_start = len(prompt_enc)
        input_ids = torch.tensor([full_enc], dtype=torch.long).to(device)
        with accumulator:
            with torch.no_grad(): outputs = model(input_ids)
            feats = accumulator.compute_features(outputs.logits, input_ids, answer_start)
        del outputs
        torch.cuda.empty_cache()
        if feats is not None:
            X_scalars.append(feats['scalars']); X_probes.append(feats['probe_vec']); Y.append(row['is_hallucination'])
    return np.array(X_scalars), np.array(X_probes), np.array(Y)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    TRAIN_PATH = "training_data_final.csv"
    TEST_PATH = "knowledge_bench_public.csv"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    accumulator = GuardianAccumulator(model)

    # --- СТРОГАЯ ФИЛЬТРАЦИЯ (УДАЛЯЕМ ШУМ) ---
    df_raw = pd.read_csv(TRAIN_PATH)
    df_true = df_raw[df_raw['method'] == 'exact'].copy()
    df_true['is_hallucination'] = 0
    df_false = df_raw[df_raw['method'] == 'number_mismatch'].copy()
    df_false['is_hallucination'] = 1
    
    # Если вранья мало, добираем из no_match
    if len(df_false) < 150:
        df_extra = df_raw[df_raw['method'] == 'no_match'].sample(n=min(200, len(df_raw[df_raw['method']=='no_match'])), random_state=42)
        df_extra['is_hallucination'] = 1
        df_train_df = pd.concat([df_true, df_false, df_extra])
    else:
        df_train_df = pd.concat([df_true, df_false])
        
    df_train_df = df_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Cleaned training set size: {len(df_train_df)}")

    # Смена имен колонок для совместимости с тестом
    df_test_raw = pd.read_csv(TEST_PATH)
    df_test_raw = df_test_raw.rename(columns={'model_answer': 'model_answer', 'response': 'model_answer', 'question': 'prompt'})

    # Экстракция
    train_scalars, train_probes, y_train = extract_dataset_features(df_train_df, model, tokenizer, accumulator, model.device)
    test_scalars, test_probes, y_test = extract_dataset_features(df_test_raw, model, tokenizer, accumulator, model.device)

    # ML
    print("Training...")
    pca = PCA(n_components=32, random_state=42)
    scaler = StandardScaler()
    
    X_train = np.hstack([train_scalars, pca.fit_transform(train_probes)])
    X_train = scaler.fit_transform(X_train)
    
    X_test = np.hstack([test_scalars, pca.transform(test_probes)])
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    
    y_score = clf.predict_proba(X_test)[:, 1]
    print(f"\n🚀 RESULT PR-AUC: {average_precision_score(y_test, y_score):.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/logreg_model.pkl")
    joblib.dump(pca, "model/pca.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
