# Resume Bullets

## Russian

- Разработал детектор фактологических галлюцинаций для GigaChat на основе hidden-state probing и uncertainty features; решение заняло 3-е место в хакатонном треке Сбера.
- Достиг PR-AUC ~0.8541 на финальном/private бенчмарке и 0.8493 на public validation без RAG, внешних API и LLM-as-a-judge на инференсе.
- Построил пайплайн признаков из одного forward pass: активации слоев L3/L9/L15, contrast directions, PCA-сжатие, logit-based uncertainty metrics и быстрый LogisticRegression-классификатор.
- Собрал offline synthetic augmentation pipeline для генерации фактологических вопросов, ответов GigaChat и верифицированных тренировочных меток, улучшив качество contrast directions.

## English

- Built a factual hallucination detector for GigaChat using hidden-state probing and uncertainty features; placed 3rd in the Sber hackathon track.
- Reached ~0.8541 PR-AUC on the final/private benchmark and 0.8493 public-validation PR-AUC without runtime RAG, external APIs, or LLM-as-a-judge.
- Designed a fast single-forward-pass feature pipeline using L3/L9/L15 activations, contrast directions, PCA compression, logit-based uncertainty metrics, and logistic regression.
- Created an offline synthetic augmentation pipeline for factual questions, GigaChat answers, and verified labels, improving contrast-direction quality.
