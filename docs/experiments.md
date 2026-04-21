# Experiment History

This project started from the competition baseline idea: use signals already available during a GigaChat forward pass instead of verifying answers with retrieval, another LLM, or external APIs. The final detector reached 3rd place in the hackathon track with a private PR-AUC of about 0.8541.

## Track Constraints

- Target: factual hallucination detection for GigaChat-generated answers.
- Main metric: PR-AUC.
- Speed mattered as a tie-breaker.
- Runtime external APIs, RAG, and LLM-as-a-judge verification were not suitable for the detector.
- One forward pass through `ai-sage/GigaChat3-10B-A1.8B-bf16` could be used to collect internal signals.

## Iteration 1: Uncertainty Baseline

The first useful baseline used answer-token logits:

- mean/min/max/std token log-probability;
- entropy statistics;
- top-1 and top-5 probabilities;
- top-1 vs top-2 margin;
- answer length and low-confidence token share.

This was fast and simple, but plateaued around PR-AUC `~0.80`.

## Iteration 2: Hidden-State Probing

The next step added hidden states from transformer layers. The detector reads representations during teacher forcing on `prompt + answer`, then trains a shallow classifier on the extracted vectors.

Late-layer probes (`L20`, `L25`) improved the baseline to `0.8202`, but were not the best signal. Early and middle layers turned out to be stronger.

## Iteration 3: Contrast Directions

For each selected layer representation, the pipeline computes a contrast direction:

```text
direction = mean(hidden_state | hallucination) - mean(hidden_state | correct)
```

The projection onto this normalized direction becomes a compact feature. This gave a strong one-dimensional signal per representation and reduced dependence on a large nonlinear classifier.

## Iteration 4: Synthetic Data Augmentation

Additional factual questions were generated offline, answered by GigaChat, and verified offline. The generated set was useful for estimating cleaner contrast directions. In experiments, the generated-data direction improved public PR-AUC to `0.8346`.

This data is not used as an external service during inference; it only expands the training signal.

## Final Configuration

The strongest public-validation setup used:

- probe layers: `L3`, `L9`, `L15`;
- first-answer-token and answer-mean representations;
- 18 uncertainty scalars;
- PCA dimensions: 32 for first-token probes, 24 for mean probes;
- logistic regression with `C=0.07`;
- mixed direction sources from original and generated data.

Public PR-AUC from `configs/best_config.json`: `0.8493`.

Final hackathon/private result: `~0.8541`, 3rd place.

## What Did Not Work As Well

- A pure uncertainty-only detector was too shallow.
- Late layers alone were weaker than early/middle layer probes.
- Heavier classifiers were not clearly worth the extra complexity for this setup.
- Runtime external validation would violate the intended speed and architecture constraints.
