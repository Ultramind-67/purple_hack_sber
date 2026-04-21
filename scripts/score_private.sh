#!/bin/bash
# scripts/score_private.sh — Скоринг private test
#
# Читает:  data/bench/knowledge_bench_private*.csv
# Создаёт: data/bench/knowledge_bench_private_scores.csv

set -e

echo "=== Scoring private test ==="
cd "$(dirname "$0")/.."

# Ищем private test файл
PRIVATE_FILE=""
for f in data/bench/knowledge_bench_private.csv \
         data/bench/knowledge_bench_private_no_labels.csv; do
    if [ -f "$f" ]; then
        PRIVATE_FILE="$f"
        break
    fi
done

if [ -z "$PRIVATE_FILE" ]; then
    echo "❌ No private test file found in data/bench/"
    exit 1
fi

echo "  Input: $PRIVATE_FILE"

export PYTHONPATH="${PYTHONPATH}:src"
python -m hallucination_detector.score_private "$PRIVATE_FILE"

echo "=== Done ==="
echo "  Output: data/bench/knowledge_bench_private_scores.csv"
