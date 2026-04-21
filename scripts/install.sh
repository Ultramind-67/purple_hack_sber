#!/bin/bash
# scripts/install.sh — Установка зависимостей и загрузка модели

set -e
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading GigaChat model (will be cached by HuggingFace) ==="
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('ai-sage/GigaChat3-10B-A1.8B-bf16', trust_remote_code=True)
print('✅ Model will be downloaded on first use')
"

echo "=== Checking data files ==="
for f in data/training/training_data_labeled.csv data/bench/knowledge_bench_public.csv; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ Missing: $f"
    fi
done

echo "=== Done ==="
