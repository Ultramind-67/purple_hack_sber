#!/bin/bash
# Train/evaluate the lightweight detector from cached feature files.
#
# This script does not regenerate GigaChat hidden-state caches. Full extraction is
# GPU-heavy and depends on the original hackathon data/artifact setup.

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"

echo "=== Training classifier from cached features ==="

for f in features/train_all_layers.npz features/gen_all_layers.npz features/test_all_layers.npz; do
    if [ ! -f "$f" ]; then
        echo "Missing cached feature file: $f"
        echo "See features/README.md for artifact notes."
        exit 1
    fi
done

python - <<'PY'
from hallucination_detector.classifier import (
    load_config,
    train_classifier,
    evaluate_on_public,
)

cfg = load_config("configs/best_config.json")
clf, scaler, pcas, directions = train_classifier(
    cfg, features_dir="features", data_dir="data"
)
score = evaluate_on_public(
    clf, scaler, pcas, directions, cfg, features_dir="features", data_dir="data"
)
print(f"Public PR-AUC: {score:.4f}")
PY

echo "=== Done ==="
