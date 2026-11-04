# Лабораторная работа 1. Диагностика обучения MLP

Ноутбук к заданию `lab1.docx`: MLP на `make_moons`, конфигурации A-F,
два seed, метрики обучения и нормы градиентов скрытых слоев.

## Требования

- Python 3.12.3
- CPU, GPU не нужен
- пакеты из `requirements.txt`

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
jupyter notebook notebook.ipynb
```

Ячейки идут сверху вниз, запускать нужно по порядку. Без браузера:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Команды выполняются из корня проекта: ноутбук пишет файлы в `results/`
и `figures/` относительно текущего каталога.

## Что создается

- `results/data_summary.json`: размеры выборок, число объектов по классам,
  средние и std признаков train после масштабирования
- `figures/scatter_train_raw.png`: диаграмма рассеяния обучающих объектов
  до масштабирования
- `results/metrics.csv`: метрики 12 запусков за эпохи 0-100
- `results/epoch100.csv`: train и val loss/accuracy на 100-й эпохе
- `results/val_loss_mean.csv`: средний val loss по двум seed
- `figures/val_loss_seed11.png`, `figures/val_acc_seed11.png`: кривые A-F
- дальше по ходу работы: `results/grad_norms.csv` и остальные графики

## Конфигурации

Ширина скрытых слоев 32, 100 эпох, batch size 128, seed 11 и 29.

| Код | Скрытые слои | Активация | Инициализация | Оптимизатор | Learning rate |
|-----|--------------|-----------|---------------|-------------|---------------|
| A   | 6            | Sigmoid   | Xavier normal | Adam        | 0.003         |
| B   | 6            | Tanh      | Xavier normal | Adam        | 0.003         |
| C   | 6            | ReLU      | Xavier normal | Adam        | 0.003         |
| D   | 6            | ReLU      | He normal     | Adam        | 0.003         |
| E   | 6            | ReLU      | He normal     | SGD         | 0.05          |
| F   | 1            | ReLU      | He normal     | Adam        | 0.003         |

У SGD стоит `momentum=0`.

## Устройство

Все эксперименты идут на CPU. В ноутбуке вызывается
`torch.set_num_threads(1)`.
