# Данные 📦

Эта папка хранит локальные датасеты хакатона и training artifacts. Для публичного GitHub лучше держать здесь только этот README и небольшие примеры, если лицензия данных это разрешает.

## Ожидаемая структура

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

## 🔐 Что не стоит публиковать

- private benchmark файлы;
- prediction scores для private benchmark;
- generated labels и synthetic training artifacts, если нет явного разрешения на публикацию;
- любые данные, полученные в рамках соревнования и не предназначенные для публичного распространения.

Код и документация остаются воспроизводимыми без публикации этих файлов: локально их можно восстановить из разрешенного источника или держать вне git.
