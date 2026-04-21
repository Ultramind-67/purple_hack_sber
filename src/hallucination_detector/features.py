"""
Feature extraction: uncertainty scalars + hidden state probes.

This is the core of the approach:
- For each (question, answer) pair, we do teacher-forcing through GigaChat
- Extract logit-based uncertainty metrics (18 scalars)
- Extract hidden states from specified layers (first answer token + mean)
"""
import torch
import numpy as np


N_UNC = 18


def setup_hooks(model, layer_indices):
    """Register forward hooks on specified layers to capture hidden states."""
    hs = {}
    hooks = []
    for idx in layer_indices:
        if idx >= len(model.model.layers):
            print(f"  ⚠️ Layer {idx} doesn't exist, skipping")
            continue

        def make_hook(i):
            def hook(mod, inp, out):
                hs[i] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        hooks.append(model.model.layers[idx].register_forward_hook(make_hook(idx)))
    return hs, hooks


def compute_uncertainty_scalars(logits, input_ids, ans_start):
    """
    Compute 18 uncertainty metrics from logits over the answer tokens.

    Features:
    0: mean log-prob
    1: min log-prob
    2: max log-prob
    3: std log-prob
    4: first token log-prob
    5: log-prob first half - second half (monotonicity)
    6: mean entropy
    7: max entropy
    8: min entropy
    9: std entropy
    10: mean top-1 probability
    11: min top-1 probability
    12: mean top-5 cumulative probability
    13: mean margin (top1 - top2)
    14: min margin
    15: perplexity (exp of negative mean log-prob)
    16: answer length (number of tokens)
    17: fraction of tokens with log-prob < -5
    """
    seq = input_ids.shape[1]
    n_ans = seq - ans_start
    if n_ans <= 1:
        return None

    al = logits[0, ans_start - 1:seq - 1].float()
    aids = input_ids[0, ans_start:seq]
    probs = torch.softmax(al, dim=-1)
    lp = torch.log_softmax(al, dim=-1)
    tlp = lp.gather(1, aids.unsqueeze(1)).squeeze(-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    t1 = probs.max(dim=-1).values
    t5 = probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1)
    t2v = probs.topk(min(2, probs.shape[-1]), dim=-1).values
    mg = t2v[:, 0] - (t2v[:, 1] if t2v.shape[1] > 1 else torch.zeros_like(t2v[:, 0]))
    mid = max(1, n_ans // 2)
    lp1 = tlp[:mid].mean().item()
    lp2 = tlp[mid:].mean().item() if n_ans > 1 else lp1

    return np.array([
        tlp.mean().item(), tlp.min().item(), tlp.max().item(),
        tlp.std().item() if n_ans > 1 else 0.0,
        tlp[0].item(), lp1 - lp2,
        ent.mean().item(), ent.max().item(), ent.min().item(),
        ent.std().item() if n_ans > 1 else 0.0,
        t1.mean().item(), t1.min().item(), t5.mean().item(),
        mg.mean().item(), mg.min().item(),
        torch.exp(-tlp.mean()).item(), float(n_ans),
        (tlp < -5.0).float().mean().item(),
    ], dtype=np.float32)


def extract_features(model, tokenizer, device, hidden_states,
                     question, answer, layer_indices):
    """
    Extract all features for a single (question, answer) pair.

    Returns dict with 'scalars' and 'L{idx}', 'L{idx}_mean' for each layer.
    """
    msgs = [{"role": "user", "content": question}]
    msgs_full = msgs + [{"role": "assistant", "content": answer}]

    enc_prompt = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt")
    prompt_ids = (enc_prompt["input_ids"] if isinstance(enc_prompt, dict)
                  else enc_prompt)
    prompt_len = prompt_ids.shape[1]

    enc_full = tokenizer.apply_chat_template(msgs_full, return_tensors="pt")
    full_ids = (enc_full["input_ids"] if isinstance(enc_full, dict)
                else enc_full).to(device)
    seq = full_ids.shape[1]
    n_ans = seq - prompt_len

    if n_ans <= 1:
        return None

    with torch.no_grad():
        logits = model(full_ids).logits

    scalars = compute_uncertainty_scalars(logits, full_ids, prompt_len)
    if scalars is None:
        del logits
        return None

    result = {'scalars': scalars}
    for idx in layer_indices:
        if idx in hidden_states:
            h = hidden_states[idx][0]
            result[f'L{idx}'] = h[prompt_len - 1].cpu().float().numpy()
            result[f'L{idx}_mean'] = h[prompt_len:seq].mean(dim=0).cpu().float().numpy()

    del logits
    return result
