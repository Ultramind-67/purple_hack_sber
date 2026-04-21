# Cached Features

The `.npz` files in this directory are local feature caches extracted from GigaChat hidden states and logits. They are useful because recomputing all activations is GPU-heavy.

Expected files:

- `train_all_layers.npz`
- `gen_all_layers.npz`
- `test_all_layers.npz`

These files can be hundreds of megabytes and should not be committed to a public GitHub repository. Keep them locally, store them in external artifact storage, or regenerate them with the training pipeline when the required data and GPU environment are available.

The classifier code expects this directory by default:

```bash
export PYTHONPATH="$PWD/src"
python -m hallucination_detector.score_private
```
