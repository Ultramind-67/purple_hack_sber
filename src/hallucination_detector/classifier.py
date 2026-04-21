"""
Classifier training and inference.

Pipeline:
1. Load training features (old labeled + generated verified)
2. Compute contrast directions from labeled data
3. PCA-compress hidden state probes
4. Project onto contrast directions
5. Combine with uncertainty scalars
6. Train LogisticRegression
"""
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def load_config(config_path="configs/best_config.json"):
    with open(config_path) as f:
        return json.load(f)


def train_classifier(config, features_dir="features", data_dir="data"):
    """
    Train the full pipeline from cached features.

    Returns: clf, scaler, pca_models, directions
    """
    cfg = config
    probe_layers = cfg['probe_layers']
    n_unc = cfg['n_uncertainty_scalars']
    pca_first = cfg['pca_first_token']
    pca_mean = cfg['pca_mean_token']
    gen_weight = cfg['gen_weight']
    dir_source = cfg['direction_source']

    # Load features
    train_f = dict(np.load(f"{features_dir}/train_all_layers.npz", allow_pickle=True))
    gen_f = dict(np.load(f"{features_dir}/gen_all_layers.npz", allow_pickle=True))

    # Load training labels
    train_dfs = []
    for f in ["training_data_labeled.csv", "training_data_labeled_hot.csv",
              "training_data_labeled_subtle.csv"]:
        fpath = os.path.join(data_dir, "training", f)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df = df[df['is_hallucination'].notna()].reset_index(drop=True)
            if 'model_answer' in df.columns:
                df = df[~df['model_answer'].astype(str).str.startswith('[ERROR')].reset_index(drop=True)
            train_dfs.append(df)
    df_all = pd.concat(train_dfs, ignore_index=True)
    train_idx = train_f['indices']
    y_train_full = df_all['is_hallucination'].values
    y_train = np.array([y_train_full[i] for i in train_idx if i < len(y_train_full)])
    n_train = len(y_train)

    # Load gen labels
    for lp in [f"{data_dir}/generated/bench_verified_labels_all.csv",
               f"{data_dir}/generated/bench_verified_labels.csv",
               "bench_verified_labels_all.csv", "bench_verified_labels.csv"]:
        if os.path.exists(lp):
            gen_labels = pd.read_csv(lp).sort_values('index').reset_index(drop=True)
            break
    n_gen = len(gen_f['indices'])
    y_gen = gen_labels['is_hallucination'].values[:n_gen]

    print(f"  Train: {n_train} ({y_train.mean():.1%}), Gen: {n_gen} ({y_gen.mean():.1%})")

    # Build combined training data
    comb_scalars = np.vstack(
        [train_f['scalars'][:n_train, :n_unc]] +
        [gen_f['scalars'][:n_gen, :n_unc]] * gen_weight
    )
    y_comb = np.concatenate([y_train] + [y_gen] * gen_weight)

    parts = [comb_scalars]
    pca_models = {}
    directions = {}

    for layer_idx in probe_layers:
        for suffix in ['', '_mean']:
            key = f'L{layer_idx}{suffix}'
            pdim = pca_mean if suffix == '_mean' else pca_first

            tr_p = np.vstack(
                [train_f[key][:n_train]] + [gen_f[key][:n_gen]] * gen_weight
            )

            nc = min(pdim, tr_p.shape[0] - 1, tr_p.shape[1])
            pca = PCA(n_components=nc, random_state=42)
            parts.append(pca.fit_transform(tr_p))
            pca_models[key] = pca

            # Direction from best source
            src = dir_source.get(key, 'gen')
            if src == 'old':
                dp, dy = train_f[key][:n_train], y_train
            else:
                dp, dy = gen_f[key][:n_gen], y_gen
            d = dp[dy == 1].mean(0) - dp[dy == 0].mean(0)
            d = d / (np.linalg.norm(d) + 1e-10)
            directions[key] = d
            parts.append((tr_p @ d).reshape(-1, 1))

    X_train = np.hstack(parts)
    print(f"  Features: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    cw = cfg.get('class_weight', None)
    clf = LogisticRegression(
        C=cfg['logistic_regression_C'],
        max_iter=5000,
        class_weight=cw,
        random_state=42
    )
    clf.fit(X_train_s, y_comb)

    return clf, scaler, pca_models, directions


def feats_to_vector(feats, pca_models, directions, probe_layers, n_unc=18):
    """Convert raw extracted features to classifier input vector."""
    parts = [feats['scalars'][:n_unc].reshape(1, -1)]

    for layer_idx in probe_layers:
        for suffix in ['', '_mean']:
            key = f'L{layer_idx}{suffix}'
            p = feats[key].reshape(1, -1)
            parts.append(pca_models[key].transform(p))
            d = directions[key]
            parts.append((p @ d).reshape(1, 1))

    return np.hstack(parts)


def evaluate_on_public(clf, scaler, pca_models, directions,
                       config, features_dir="features", data_dir="data"):
    """Evaluate pipeline on public test."""
    probe_layers = config['probe_layers']
    n_unc = config['n_uncertainty_scalars']

    pub_f = dict(np.load(f"{features_dir}/test_all_layers.npz", allow_pickle=True))
    pub_df = pd.read_csv(f"{data_dir}/bench/knowledge_bench_public.csv")
    pub_idx = pub_f['indices']
    y_pub = pub_df['is_hallucination'].values[pub_idx]
    n_pub = len(y_pub)

    parts = [pub_f['scalars'][:n_pub, :n_unc]]
    for layer_idx in probe_layers:
        for suffix in ['', '_mean']:
            key = f'L{layer_idx}{suffix}'
            ep = pub_f[key][:n_pub]
            parts.append(pca_models[key].transform(ep))
            d = directions[key]
            parts.append((ep @ d).reshape(-1, 1))

    X_pub = np.hstack(parts)
    pred = clf.predict_proba(scaler.transform(X_pub))[:, 1]
    prauc = average_precision_score(y_pub, pred)
    return prauc
