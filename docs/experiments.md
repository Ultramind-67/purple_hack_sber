# История экспериментов 🧪

Проект вырос из идеи baseline-решения для хакатона: использовать сигналы, которые уже доступны во время forward pass GigaChat, вместо внешней проверки через поиск, другую LLM или API. Финальный детектор занял 3 место в треке и показал около `0.8541` PR-AUC на private/final бенчмарке.

## 🎯 Ограничения трека

- Цель: детекция фактологических галлюцинаций в ответах GigaChat.
- Главная метрика: PR-AUC.
- Скорость учитывалась как tie-breaker.
- Runtime external APIs, RAG и LLM-as-a-judge не подходили для финального детектора.
- Один forward pass через `ai-sage/GigaChat3-10B-A1.8B-bf16` можно было использовать для сбора внутренних сигналов.

## 1. Uncertainty baseline

Первый полезный baseline строился на logits токенов ответа:

- mean/min/max/std token log-probability;
- entropy statistics;
- top-1 и top-5 probabilities;
- margin между top-1 и top-2;
- длина ответа и доля low-confidence токенов.

Подход был быстрым и простым, но упирался примерно в PR-AUC `~0.80`.

## 2. Hidden-state probing

Следующий шаг — добавить hidden states из transformer layers. Детектор читает представления при teacher forcing на `prompt + answer`, затем обучает небольшой классификатор поверх извлеченных векторов.

Late-layer probes (`L20`, `L25`) улучшили baseline до `0.8202`, но оказались не лучшим источником сигнала. Более сильными стали ранние и средние слои.

## 3. Contrast directions

Для каждого выбранного layer representation считается contrast direction:

```text
direction = mean(hidden_state | hallucination) - mean(hidden_state | correct)
```

Проекция на нормированное направление становится компактным признаком. Это дало сильный одномерный сигнал на каждое представление и снизило зависимость от тяжелого nonlinear-классификатора.

## 4. Synthetic data augmentation

Дополнительные фактологические вопросы генерировались offline, затем на них отвечал GigaChat, а корректность ответов проверялась offline-разметкой. Generated set оказался полезен для оценки более чистых contrast directions.

На экспериментах generated-data direction поднял public PR-AUC до `0.8346`.

Важно: эти данные не используются как внешний сервис во время inference. Они только расширяют training signal.

## 🏁 Финальная конфигурация

Лучшая public-validation конфигурация:

- probe layers: `L3`, `L9`, `L15`;
- first-answer-token и answer-mean representations;
- 18 uncertainty scalars;
- PCA dimensions: 32 для first-token probes, 24 для mean probes;
- logistic regression с `C=0.07`;
- mixed direction sources из original и generated data.

Public PR-AUC из `configs/best_config.json`: `0.8493`.

Финальный/private результат: `~0.8541`, 3 место.

## Что сработало хуже

- Pure uncertainty-only detector был слишком слабым.
- Late layers сами по себе оказались хуже early/mid layer probes.
- Более тяжелые классификаторы не дали очевидного выигрыша относительно сложности.
- Runtime external validation противоречил ограничениям по скорости и архитектуре решения.
