# Guardian of Truth: детекция галлюцинаций через hidden-state probing 🧠

Конкурсное ML-решение для определения фактологических галлюцинаций в ответах GigaChat. Детектор использует внутренние сигналы модели из одного forward pass: hidden states, contrast directions, PCA-сжатые представления и uncertainty-признаки из logits.

**Результат на хакатоне:** 3 место в треке Сбера. 🥉  
**Final/private PR-AUC:** ~0.8541.  
**Public validation PR-AUC:** 0.8493 по конфигу `configs/best_config.json`.

## ✨ Почему проект интересный

[Задача состояла в том](task.md), чтобы предсказать, содержит ли ответ `ai-sage/GigaChat3-10B-A1.8B-bf16` фактологическую галлюцинацию. Важным ограничением была скорость: можно было использовать forward pass самой GigaChat-модели для снятия внутренних сигналов, но нельзя было строить runtime-решение на внешних API, RAG или LLM-as-a-judge.

Финальный подход рассматривает детекцию галлюцинаций как задачу анализа внутренних представлений модели:

- снимаем hidden states с ранних и средних слоев при teacher forcing;
- считаем uncertainty-признаки по logits токенов ответа;
- строим contrast directions между корректными и галлюцинированными ответами;
- сжимаем probe-векторы через PCA;
- обучаем легкий `LogisticRegression`-классификатор для быстрого inference.

## 🔬 Метод

1. **Teacher forcing**  
   Подаем в GigaChat пару `prompt + model_answer` и читаем активации на span ответа.

2. **Hidden-state probes**  
   Используем слои `L3`, `L9` и `L15`. Ранние слои оказались особенно полезными: рабочая гипотеза была в том, что фактологический сигнал появляется раньше, чем поздние reasoning/generation-слои успевают его исказить.

3. **Contrast directions**  
   Для каждого probe-представления считаем нормированную разницу средних активаций:

   ```text
   direction = mean(hidden_state | hallucination) - mean(hidden_state | correct)
   ```

   Проекция на это направление становится компактным и сильным признаком.

4. **Uncertainty scalars**  
   Добавляем 18 признаков из logits: статистики log-probability, entropy, top-k probabilities, margin, длина ответа, proxy perplexity и доля low-confidence токенов.

5. **Synthetic training data**  
   Дополнительные фактологические вопросы генерировались offline, затем GigaChat отвечал на них, а разметка проверялась offline. Эти данные используются только для обучения, не для runtime-инференса.

6. **Lightweight classifier**  
   Объединяем scalars, PCA-сжатые probes и contrast projections, после чего обучаем `LogisticRegression`.

## 📈 Результаты

| Эксперимент | Public PR-AUC |
| --- | ---: |
| Baseline только на uncertainty | ~0.80 |
| Late-layer probes (`L20`, `L25`) | 0.8202 |
| Contrast direction на generated data | 0.8346 |
| Early/mid probes (`L3`, `L9`, `L15`) | 0.8493 |
| Финальный хакатонный submission | ~0.8541 private |

Подробнее про ход экспериментов: [`docs/experiments.md`](docs/experiments.md).

## 📁 Структура репозитория

```text
.
├── configs/
│   ├── best_config.json          # Лучшая public-validation конфигурация
│   └── config.json               # Более ранний компактный конфиг для run.py
├── data/
│   ├── README.md                 # Политика данных и ожидаемая структура
│   ├── bench/                    # Public/private benchmark CSV
│   ├── generated/                # Offline synthetic data и labels
│   └── training/                 # Labeled training splits
├── docs/
│   ├── experiments.md            # История экспериментов
│   └── resume.md                 # Формулировки для резюме
├── features/
│   ├── README.md                 # Описание feature-кэшей
│   ├── train_all_layers.npz
│   ├── gen_all_layers.npz
│   └── test_all_layers.npz
├── scripts/
│   ├── install.sh
│   ├── train.sh
│   └── score_private.sh
├── src/hallucination_detector/
│   ├── classifier.py             # PCA, contrast directions, обучение классификатора
│   ├── features.py               # Logit и hidden-state feature extraction
│   ├── patches.py                # Патчи загрузки GigaChat / DeepSeek-V3
│   └── score_private.py          # End-to-end scoring entry point
├── run.py                        # Single-file hackathon runner
└── requirements.txt
```

## ⚙️ Быстрый старт

Рекомендуется Python 3.10+ и CUDA GPU. На хакатоне использовался сервер с RTX 5090 32 GB, финальная проверка ориентировалась на A100.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Если feature-кэши уже лежат в `features/`, можно обучить и проверить легкий классификатор без повторного снятия всех GigaChat-активаций:

```bash
bash scripts/train.sh
```

Для скоринга private benchmark:

```bash
bash scripts/score_private.sh
```

Полное пересчитывание features требует загрузки `ai-sage/GigaChat3-10B-A1.8B-bf16` и достаточно тяжелого GPU-окружения. Текущий public-facing репозиторий сохраняет стабильный путь обучения/скоринга по кэшированным признакам и документирует артефакты. Перед регенерацией см. [`features/README.md`](features/README.md).

