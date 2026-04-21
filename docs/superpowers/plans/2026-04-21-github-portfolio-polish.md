# GitHub Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the hackathon snapshot into a GitHub-ready portfolio repository.

**Architecture:** Keep the ML pipeline unchanged and add documentation, dependency metadata, and publishing hygiene around it. The main README explains the project; supporting docs capture experiment history, artifact policy, and resume wording.

**Tech Stack:** Python, PyTorch, Transformers, scikit-learn, pandas, NumPy, shell scripts.

---

## File Structure

- Modify `README.md`: portfolio-facing project overview, results, method, structure, quick start, publishing notes.
- Modify `.gitignore`: ignore local shell files, caches, model/checkpoint outputs, and large generated artifacts.
- Create `requirements.txt`: runtime dependencies used by scripts.
- Create `docs/experiments.md`: concise experiment history and leaderboard result.
- Create `docs/resume.md`: resume bullets in Russian and English.
- Create `data/README.md`: dataset expectations and public-sharing policy.
- Create `features/README.md`: cached feature artifact explanation.

## Tasks

- [ ] Update `.gitignore` with repository hygiene rules.
- [ ] Add `requirements.txt` matching imports used in the current scripts.
- [ ] Rewrite `README.md` around the 3rd-place result and reproducible pipeline.
- [ ] Add `docs/experiments.md` with the progression from uncertainty baseline to hidden-state contrast probes.
- [ ] Add `docs/resume.md` with concise bullets for CV and GitHub profile use.
- [ ] Add artifact README files under `data/` and `features/`.
- [ ] Run `python -m compileall -q src run.py`.
- [ ] Run `git status --short` and summarize remaining publish risks.
