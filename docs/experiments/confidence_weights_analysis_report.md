# Анализ проблемы с confidence_weights

## Суть фильтра confidence_weights

Фильтр `confidence_weights` используется для оценки качества меток в датасете и фильтрации "неопределённых" примеров.

### Как вычисляется label_confidence

Функция `compute_label_confidence()` в [`dataset_builder_v2.py`](src/paraglideml/data/dataset_builder_v2.py:20) вычисляет уверенность в метке:

**Для is_flyable = 1 (лётный день):**
- `flight_count >= 5`: confidence = 1.0 (точно лётный)
- `flight_count 3-4`: confidence = 0.9

**Для is_flyable = 0 (нелётный день):**
- `flight_count = 0` и `weekend`: confidence = 0.8 (вероятно нелётный)
- `flight_count = 0` и `weekday`: confidence = 0.5 (неопределённо)
- `flight_count 1-2`: confidence = 0.6 (пограничный случай)

### Как работает фильтрация

Фильтр удаляет примеры с confidence ниже заданного порога:

- **Threshold >= 0.5**: 9,300 samples, flyable ratio: 63.9%
- **Threshold >= 0.6**: 7,561 samples, flyable ratio: 78.6%
- **Threshold >= 0.7**: 6,337 samples, flyable ratio: 93.8%
- **Threshold >= 0.8**: 6,337 samples, flyable ratio: 93.8%
- **Threshold >= 0.9**: 5,946 samples, flyable ratio: 100.0%

## Проблема с экспериментом 005

### Почему возникает дисбаланс

1. **Исходный датасет**: 9,300 samples, 63.9% flyable
2. **При фильтре >= 0.7**: удаляется 2,963 samples
3. **Удаляются в основном будние дни с 0 полётами** (confidence = 0.5)
4. **Остаются**:
   - Flyable дни (confidence 0.9-1.0)
   - Not flyable выходные (confidence 0.8)
5. **Результат**: 93.8% flyable, сильный дисбаланс

### Конкретные цифры

- **Будние дни, not flyable**: 2,728 samples (confidence = 0.5)
- **Выходные дни, not flyable**: 626 samples (confidence = 0.6-0.8)
- **При фильтре >= 0.7**: удаляются все будние not flyable дни
- **Test set**: 95% flyable (1244 vs 65)

## Почему Macro F1 не подходит

Macro F1 усредняет F1 по классам, что не учитывает дисбаланс. При 95% flyable классе:

- Модель может просто предсказывать "flyable" и получать высокий Macro F1
- Но это не отражает реальную способность модели различать классы
- Все ячейки показывают F1 > 0.87, но это из-за дисбаланса

## Решения проблемы

### 1. Использовать более низкий порог фильтрации
- **Threshold >= 0.6**: 78.6% flyable, более сбалансированный
- Сохраняет больше not flyable примеров

### 2. Использовать confidence-weighted loss без фильтрации
- В [`train_v3.py`](src/paraglideml/train_v3.py:198): `USE_CONFIDENCE_WEIGHTING = True`
- Веса confidence используются в loss функции, но не для фильтрации
- Сохраняет все данные, но учитывает качество меток

### 3. Разделение train/test по времени
- Разделять по годам, а не случайно
- Избегать утечки данных между train и test
- Более реалистичная оценка производительности

### 4. Balanced sampling или class weights
- Использовать balanced batch sampling
- Добавить class weights в loss функцию
- Учитывать дисбаланс классов

## Рекомендации

1. **Не использовать фильтрацию >= 0.7** для test set
2. **Использовать confidence-weighted loss** без фильтрации
3. **Разделять данные по времени** (train: 2021-2024, test: 2025)
4. **Использовать balanced metrics** (Balanced Accuracy, F1-score с учетом дисбаланса)
5. **Анализировать per-cell performance** отдельно от общего F1

## Заключение

Фильтр confidence_weights полезен для оценки качества меток, но его использование для фильтрации test set создаёт сильный дисбаланс. Лучше использовать confidence weights в loss функции для обучения, но не для фильтрации данных при оценке.