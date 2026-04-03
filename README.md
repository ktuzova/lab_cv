# Лабораторная работа 1: CV — Классификация рентгеновских снимков (пневмония). Задание на пятерку

## Описание задачи

Бинарная классификация рентгеновских снимков грудной клетки: **NORMAL** (норма) vs **PNEUMONIA** (пневмония).

Практическое применение — автоматический скрининг пневмонии как ассистент рентгенолога в телемедицине и клинической диагностике.

## Датасет

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5856 рентгеновских снимков грудной клетки, собранных в Guangzhou Women and Children's Medical Center.

| Split | NORMAL | PNEUMONIA | Всего |
|-------|--------|-----------|-------|
| Train | 1072   | 3100      | 4172  |
| Val   | 269    | 775       | 1044  |
| Test  | 234    | 390       | 624   |

Валидационная выборка пересоздана (20% от train), т.к. оригинальная содержала всего 16 изображений.

## Метрики

- **Recall** (главная) — в медицинской задаче критично не пропустить больного
- **F1-score** — баланс precision/recall
- **Accuracy** — общая корректность
- **ROC-AUC** — оценка модели независимо от порога

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| ResNet-18 baseline | 0.696 | 0.672 | 1.000 | 0.804 | 0.878 |
| ResNet-50 baseline | 0.771 | 0.733 | 0.997 | 0.845 | 0.848 |
| EfficientNet-B0 baseline | 0.827 | 0.787 | 0.992 | 0.878 | 0.945 |
| ViT-B/16 baseline | 0.753 | 0.718 | 0.997 | 0.835 | 0.955 |
| **ResNet-18 improved** | **0.901** | **0.866** | **0.995** | **0.926** | **0.976** |
| ResNet-50 improved | 0.875 | 0.836 | 0.995 | 0.909 | 0.971 |
| EfficientNet-B0 improved | 0.872 | 0.834 | 0.992 | 0.906 | 0.979 |
| **ViT-B/16 improved** | **0.904** | **0.872** | **0.992** | **0.928** | **0.982** |
| Custom CNN baseline | 0.731 | 0.699 | 1.000 | 0.823 | 0.941 |
| Custom CNN improved | 0.859 | 0.827 | 0.980 | 0.897 | 0.941 |

## Структура репозитория

```
chest-xray-classification/
├── README.md
├── requirements.txt
└── lab1.ipynb          # ноутбук с полным пайплайном
```

## Инструкция по запуску

### Вариант 1: Google Colab (рекомендуется)

1. Откройте ноутбук в Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/chest-xray-classification/blob/main/lab1.ipynb)
2. Подключите GPU: **Runtime → Change runtime type → T4 GPU**
3. При запросе введите Kaggle credentials (username + API key из [kaggle.com/settings](https://www.kaggle.com/settings))
4. Выполните все ячейки последовательно (**Runtime → Run all**)

### Вариант 2: Локальный запуск

```bash
git clone https://github.com/USERNAME/chest-xray-classification.git
cd chest-xray-classification
pip install -r requirements.txt
jupyter notebook lab1.ipynb
```

Требуется GPU с поддержкой CUDA. Для скачивания датасета необходим файл `kaggle.json`.

## Зависимости

- Python 3.10+
- PyTorch 2.x
- torchvision
- scikit-learn
- matplotlib
- pandas
- tqdm
- opendatasets

## Методология

1. **EDA** — анализ распределения классов, визуализация примеров
2. **Бейзлайн** — обучение pretrained моделей (ResNet-18, ResNet-50, EfficientNet-B0, ViT-B/16) из torchvision с базовыми трансформациями
3. **Улучшение бейзлайна** — аугментации (flip, rotation, affine, color jitter), балансировка классов (WeightedRandomSampler), CosineAnnealingLR scheduler
4. **Кастомная модель** — CNN из 5 свёрточных блоков (Conv-BN-ReLU-Pool), реализованная с нуля, обучена с и без улучшений

## Основные выводы

- Transfer learning значительно превосходит обучение с нуля
- Балансировка классов — ключевой фактор на несбалансированных данных (Accuracy +13–20%)
- Аугментации и scheduler дают дополнительный прирост качества
- Лучшая модель — ViT-B/16 improved (F1=0.928, ROC-AUC=0.982)
