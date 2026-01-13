# Сравнение train_v3_fixed.py и src/paraglideml/train_v4.py

## Ключевые отличия

### 1. Фильтрация по confidence

**train_v3_fixed.py:**
- `FILTER_CONFIDENCE = False` - по умолчанию не фильтрует
- Но есть параметр `CONFIDENCE_THRESHOLD = 0.7` для включения фильтрации
- Можно включить фильтрацию, изменив `FILTER_CONFIDENCE = True`

**train_v4.py:**
- Полностью убрана фильтрация по confidence
- Комментарий: `# NO confidence filtering - use all samples for better evaluation`
- Используются все данные без фильтрации

### 2. Baseline F1

**train_v3_fixed.py:**
- `BASELINE_F1 = 0.807` - baseline от single-cell Kobala (exp_003)

**train_v4.py:**
- `BASELINE_F1 = 0.773` - baseline от multi-cell exp_004
- Обновлено для сравнения с более новыми экспериментами

### 3. Разделение данных

**train_v3_fixed.py:**
- NEW: Разделение по времени (train: 2021-2024, test: 2025)
- NEW: Validation split по времени (август-сентябрь)
- Использует `np.where()` для индексации

**train_v4.py:**
- Стандартное случайное разделение
- `val_size = int(0.2 * len(X_train))`
- `indices = np.random.permutation(len(X_train))`

### 4. Дополнительные метрики

**train_v3_fixed.py:**
- NEW: Balanced Accuracy
- NEW: Вывод class balance для train/test
- NEW: Сохранение balanced accuracy в report.txt

**train_v4.py:**
- Только стандартные метрики (Macro F1)
- Нет balanced accuracy

### 5. Конфигурация и логирование

**train_v3_fixed.py:**
- Нет сохранения config.json
- Простое логирование

**train_v4.py:**
- Сохраняет config.json с параметрами эксперимента
- Более подробное логирование параметров

## Соответствие рекомендациям

### ✅ train_v4.py соответствует рекомендациям:

1. **Не использует фильтрацию >= 0.7** - ✅ Полностью убрана
2. **Использует confidence-weighted loss** - ✅ `USE_CONFIDENCE_WEIGHTING = True`
3. **Разделяет данные по времени** - ❌ Использует случайное разделение
4. **Использует balanced metrics** - ❌ Нет balanced accuracy
5. **Анализирует per-cell performance** - ✅ Есть per-cell analysis

### ✅ train_v3_fixed.py соответствует рекомендациям:

1. **Не использует фильтрацию >= 0.7** - ✅ `FILTER_CONFIDENCE = False` по умолчанию
2. **Использует confidence-weighted loss** - ✅ `USE_CONFIDENCE_WEIGHTING = True`
3. **Разделяет данные по времени** - ✅ NEW: Разделение по годам
4. **Использует balanced metrics** - ✅ NEW: Balanced Accuracy
5. **Анализирует per-cell performance** - ✅ Есть per-cell analysis

## Выводы

**train_v4.py** - это улучшенная версия, которая:
- Убрала проблемную фильтрацию по confidence
- Обновила baseline для актуального сравнения
- Сохранила confidence-weighted loss

**train_v3_fixed.py** - это моя улучшенная версия, которая:
- Реализует все рекомендации из анализа
- Добавляет временное разделение данных
- Добавляет balanced metrics
- Сохраняет гибкость (можно включить фильтрацию если нужно)

## Рекомендации

**train_v3_fixed.py** лучше соответствует рекомендациям, так как:
1. Реализует временное разделение данных (более реалистичная оценка)
2. Добавляет balanced accuracy (учитывает дисбаланс классов)
3. Сохраняет гибкость настройки

**train_v4.py** хорош, но можно улучшить:
1. Добавить временное разделение данных
2. Добавить balanced accuracy
3. Обновить baseline до актуального значения