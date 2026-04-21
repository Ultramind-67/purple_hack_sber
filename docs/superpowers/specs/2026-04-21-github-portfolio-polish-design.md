# GitHub Portfolio Polish Design

## Goal

Prepare the hackathon hallucination detection project for GitHub and resume use without changing the validated ML approach.

## Scope

The work is documentation and repository hygiene around an existing solution. The code should keep its current behavior: hidden-state probes from GigaChat, uncertainty scalars, contrast directions, PCA, and logistic regression. The public-facing materials should explain why the approach is interesting, how it performed, and how another engineer can understand or reproduce it.

## Public Positioning

The project should be presented as a competitive ML solution for factual hallucination detection in GigaChat responses. It should mention:

- 3rd place in the hackathon track.
- Final/private PR-AUC around 0.8541.
- Public validation PR-AUC 0.8493 from `configs/best_config.json`.
- No RAG, no runtime external APIs, no second LLM judge in the detector.

## Repository Changes

- Rewrite `README.md` as the main portfolio entry point.
- Add `requirements.txt` so install scripts are not broken.
- Add `docs/experiments.md` with a compact experiment history.
- Add `docs/resume.md` with reusable resume bullets.
- Add `data/README.md` and `features/README.md` to explain what artifacts are expected and what should not be pushed publicly.
- Strengthen `.gitignore` for caches, shell files, large local artifacts, model checkpoints, and generated outputs.

## Data And Artifact Policy

The repository currently contains data and feature caches from the hackathon snapshot. For public GitHub, the README should explicitly recommend removing private/large artifacts from the index before publishing, while keeping instructions for regenerating or restoring them locally.

## Verification

Because this is mostly packaging, verification should include:

- Python syntax compilation for `src` and `run.py`.
- A lightweight import check where possible.
- `git status --short` review so the final answer can tell the user exactly what changed.
