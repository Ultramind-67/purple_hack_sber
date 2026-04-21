# Data

This directory stores local hackathon datasets and generated training artifacts.

Expected layout:

```text
data/
├── bench/
│   ├── knowledge_bench_public.csv
│   ├── knowledge_bench_private.csv
│   └── knowledge_bench_private_scores.csv
├── generated/
│   ├── bench_questions_generated.csv
│   ├── bench_gigachat_answers.csv
│   └── bench_verified_labels_all.csv
└── training/
    ├── training_data_labeled.csv
    ├── training_data_labeled_hot.csv
    └── training_data_labeled_subtle.csv
```

Do not publish private benchmark files, generated labels, or competition data unless the data license explicitly allows it. The repository keeps code and documentation reproducible without requiring those files to be public.

For a public GitHub release, prefer keeping only this README plus any clearly redistributable sample data.
