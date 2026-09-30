# Дулько Денис Андреевич, ИИ-25, вариант 5

Лабораторные по дисциплине «Обработка изображений в интеллектуальных системах».
Преподаватель: Андренко К. В.

- ЛР 1: своя CNN, размеченная часть STL-10 (96×96), SGD.
- ЛР 2: DenseNet121 с весами ImageNet, новый классификатор на 10 классов, SGD.
- Отчёты: `lab1/rep/lab1.pdf` и `lab2/rep/lab2.pdf`.
- Графики, измерения и обученные веса: `lab1/results` и `lab2/results`.

## Запуск

Команды выполняются из корня репозитория `IP_AI_25`. Нужен Python 3.12.
Проект можно открыть в Cursor или VS Code.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r reports/Dulko/requirements.txt
```

CUDA используется автоматически при наличии совместимой видеокарты NVIDIA.
На компьютере без NVIDIA установите обычные пакеты через
`python -m pip install -r reports/Dulko/requirements.txt`; обучение на CPU дольше.
При первом обучении STL-10 и веса ImageNet скачиваются автоматически.
Архив STL-10 около 2.6 ГБ; неразмеченная часть в обучении не используется.

### Обучение

```powershell
.\.venv\Scripts\python.exe reports/Dulko/lab1/src/lab1.py
.\.venv\Scripts\python.exe reports/Dulko/lab2/src/lab2.py
```

Параметр `--epochs` задаёт число эпох, `--workers 0` отключает фоновые загрузчики
данных, если среда не поддерживает запуск дополнительных процессов.

### Проверка своего изображения

Сохранённые веса позволяют выполнять предсказание без повторного обучения.
DenseNet121 сохраняет только обученный классификатор; базовые веса ImageNet
автоматически загружаются из кэша torchvision или скачиваются при первом запуске.

```powershell
.\.venv\Scripts\python.exe reports/Dulko/lab1/src/lab1.py --image reports/Dulko/examples/dog.jpg
.\.venv\Scripts\python.exe reports/Dulko/lab2/src/lab2.py --image reports/Dulko/examples/dog.jpg
```

Вместо пути можно указать `--choose`, чтобы выбрать файл в окне.
Первая команда сохраняет `lab1/results/custom_prediction.png`, вторая сравнивает
обе сети на одном изображении и сохраняет `lab2/results/comparison.png`.
Откройте PNG, чтобы увидеть изображение, класс и значение softmax.
Softmax является оценкой модели и не гарантирует правильность ответа.
Распознаются только 10 классов STL-10:
airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.

## Как проведено сравнение

Из 5000 размеченных обучающих изображений 4000 используются для обучения,
1000 для валидации (по 400/100 изображений каждого класса, seed=5).
Обе сети используют одинаковые индексы. Лучшие веса выбираются по минимальной
ошибке на валидации. Все 8000 тестовых изображений используются только для
итоговой оценки, а не для выбора эпохи. Кривые показывают train и validation.

CNN обучается с нуля 25 эпох. У DenseNet121 заморожены признаки и статистика
BatchNorm, обучается линейный слой `1024 → 10` в течение 8 эпох.
Для обеих моделей: SGD, начальный lr=0.01, momentum=0.9,
weight_decay=0.0001, CosineAnnealingLR, CrossEntropyLoss, batch_size=64.

## Измеренные результаты

| Модель | Точность на 8000 тестовых изображениях | CrossEntropyLoss | Выбранная эпоха |
|---|---:|---:|---:|
| CNN | 52.4875% | 1.2763 | 24 из 25 |
| DenseNet121 | 96.4000% | 0.1098 | 7 из 8 |

Результаты получены на RTX 3070. На внешней фотографии `dog.jpg` обе сети
ошибочно ответили `cat`; этот пример сохранён в отчётах. Он показывает, что
тестовая точность не гарантирует правильный ответ на любую фотографию.

## Источники

- [STL-10 и протокол оценки](https://cs.stanford.edu/~acoates/stl10/).
- [STL10 в torchvision](https://docs.pytorch.org/vision/0.23/generated/torchvision.datasets.STL10.html).
- [DenseNet121 и предобработка весов](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.densenet121.html).
- [MixMatch, NeurIPS 2019, таблица 2](https://papers.nips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf): ошибка 5.59% (точность 94.41%) при 5000 метках и использовании неразмеченных данных.
- [Reduction of Class Activation Uncertainty with Background Information, таблица III](https://arxiv.org/abs/2305.03238): ViT-L/16 + Spinal FC + Background, 99.71 ± 0.06%; предобучение и дополнительные фоновые изображения. Авторы обозначают этот результат как SOTA на май 2023 года; здесь он служит опубликованным ориентиром, а не утверждением о текущем рекорде.
- [Источник демонстрационной фотографии собаки](https://github.com/pytorch/hub/blob/master/images/dog.jpg).

Числа из исследований получены при других архитектурах, данных и протоколах.
Их нельзя напрямую приравнивать к условиям этих лабораторных работ.
