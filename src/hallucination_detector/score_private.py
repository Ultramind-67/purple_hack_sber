"""
Score private test: load model, extract features, classify.
"""
import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from tqdm import tqdm

# Apply patches before importing model classes
from hallucination_detector.patches import apply_patches
apply_patches()

from transformers import AutoModelForCausalLM, AutoTokenizer
from hallucination_detector.features import setup_hooks, extract_features, N_UNC
from hallucination_detector.classifier import (
    load_config, train_classifier, feats_to_vector, evaluate_on_public
)


def main():
    # ─── Config ───
    config = load_config("configs/best_config.json")
    probe_layers = config['probe_layers']

    # ─── Find input file ───
    private_path = None
    candidates = [
        "data/bench/knowledge_bench_private.csv",
        "data/bench/knowledge_bench_private_no_labels.csv",
    ]
    if len(sys.argv) > 1:
        candidates.insert(0, sys.argv[1])

    for c in candidates:
        if os.path.exists(c):
            private_path = c
            break

    if private_path is None:
        for f in os.listdir("data/bench/"):
            if "private" in f.lower() and f.endswith(".csv"):
                private_path = f"data/bench/{f}"
                break

    if private_path is None:
        print("❌ No private test file found!")
        sys.exit(1)

    df = pd.read_csv(private_path)
    print(f"Input: {private_path} ({len(df)} rows)")

    q_col = next((c for c in ['prompt', 'question'] if c in df.columns), None)
    a_col = next((c for c in ['model_answer', 'response', 'answer'] if c in df.columns), None)
    has_answers = a_col and df[a_col].notna().sum() > 0

    # ─── Train classifier ───
    clf, scaler, pca_models, directions = train_classifier(
        config, features_dir="features", data_dir="data"
    )

    # ─── Verify on public test ───
    if os.path.exists("features/test_all_layers.npz"):
        prauc = evaluate_on_public(clf, scaler, pca_models, directions,
                                    config, features_dir="features", data_dir="data")
        print(f"  Public test PR-AUC: {prauc:.4f}")

    # ─── Load model ───
    print("Loading GigaChat model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)
    device = next(model.parameters()).device
    hidden_states, hooks = setup_hooks(model, probe_layers)

    # ─── Process ───
    predictions = []
    errors = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        question = str(row[q_col])

        if has_answers and pd.notna(row[a_col]) and str(row[a_col]).strip():
            answer = str(row[a_col])
        else:
            msgs = [{"role": "user", "content": question}]
            enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                 return_tensors="pt")
            pid = (enc["input_ids"] if isinstance(enc, dict) else enc).to(device)
            with torch.no_grad():
                out = model.generate(pid, max_new_tokens=150,
                                      temperature=0.01, do_sample=True,
                                      repetition_penalty=1.1)
            answer = tokenizer.decode(out[0, pid.shape[1]:], skip_special_tokens=True).strip()

        try:
            feats = extract_features(model, tokenizer, device,
                                      hidden_states, question, answer, probe_layers)
            if feats is not None:
                x = feats_to_vector(feats, pca_models, directions, probe_layers)
                prob = clf.predict_proba(scaler.transform(x))[0, 1]
            else:
                prob = 0.5
                errors += 1
        except Exception as e:
            print(f"  Error {i}: {e}")
            prob = 0.5
            errors += 1

        predictions.append(prob)
        if i % 50 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # ─── Save ───
    output = pd.DataFrame({'predict_proba': predictions})
    output_path = "data/bench/knowledge_bench_private_scores.csv"
    os.makedirs("data/bench", exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Saved {len(predictions)} predictions to {output_path}")
    print(f"   Mean: {np.mean(predictions):.3f}, Errors: {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
