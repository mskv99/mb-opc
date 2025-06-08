# Применение методов глубокого обучения для коррекции эффектов оптической близости в проекционной литографии

---

## Введедние

В современной фотолитографии коррекция эффектов оптической близости (Optical
Proximity Correction, OPC) играет ключевую роль в обеспечении точного
воспроизведения топологических элементов на подложке. Традиционные методы
**OPC**, основанные на физических моделях, требуют значительных вычислительных и
временных затрат. С развитием глубокого обучения появились новые подходы,
способные повысить эффективность и точность **OPC**. Одним из таких подходов
является использование моделей компьютерного зрения для решения задач попарного
переноса изображений (**paired image-to-image translation**)[1]. В таком случае
задачу **OPC** можно представить как задачу обучения с учителем, где модель
глубокой нейронной сети стремится генерировать изображения откорректированной
топологии для соответствующих изображений исходного дизайна, максимально похожие
на изображения коррекций из обучающей выборки. Описанный подход был использован
в данной работе.

## Датасет

Для создания датасета использовался набор топологических элементов из слоя
металлов с характерными размерами **32 нм**[1], дополненный собственной
откорректированной топологией. Топологические файлы были преобразованы в
бинарные изображения с разрешением **1024x1024 px**, где каждый пиксель
соответствует области **1нм2**.

## Модели

В исследовании рассматривается применение различных архитектур глубоких
нейронных сетей для коррекции эффектов оптической близости, включая модели
**DAMO** (Deep Agile Mask Optimization)[2], **CFNO** (Convolutional Fourier
Neural Operator)[3], **PSPnet** (Pyramid Scene Parsing Network), **MANet**
(Multi-Attention Network), **UperNet**.

## 🛠️ Настройка

```bash
# клонируем репозитрий
git clone https://github.com/mskv99/mb-opc.git
cd mb-opc

# создаём виртуальное окружение с помощью conda
conda create -n myenv python=3.10
conda activate myenv

# скачиваем poetry для управления зависимостями
pip install poetry
# будем использовать poetry только для управления зависимостями
poetry config virtualenvs.create false
# установим необходимые пакеты
poetry install

# подгрузим датасет для обучения с помощью dvc,
# может занять какое-то время, в датасете около 12К изображений (уменьшенная версия)
# по завершению загрузки датасет будет расположен в `data/processed/gds_dataset`
dvc pull -r gdrive_data data/processed/gds_dataset.dvc

# подгрузим веса одной из обученных моделей с помощью dvc:
# по завершению загрузки веса будут расположен в `checkpoints/upernet.ckpt`
dvc pull -r gdrive_models checkpoints/upernet.ckpt.dvc

# убедимся в отсутсвии досадных ошибок:
pre-commit run --all
```

## 🏋️ Обучение

1. **Стандартный вызов**:

```bash
# из корневой директории проекта запускаем скрипт обучения:
python src/train.py
```

В качестве CLI для обучения используем **Hydra**. Соотвествующие конфиги можно
находятся в папке `configs`.

Основная группа гиперпараметров:

- `model` - выбираем доступную из модель из списка доступных [`unet`, `cfno`,
  `pspnet`, `manet`, `upernet`]. По умолчанию: `UNet` без добавления
  `skip-connections`
- `optim` - тип оптимизатора. По умолчанию: `Adam`.
- `lr` - learning rate. По умолчанию: `2e-4`
- `weight_decay` - weight_decay. По умолчанию: `1e-5`
- `sched` - тип планировщика. По умолчанию: `CosineAnnealing`
- `eta_min` - значение learning rate в конце обучения. По умолчанию: `1e-7`
- `epochs` - число эпох обучения. По умолчанию: `50`
- `batch_size` - размер батча. По умолчанию: `6`
- `device` - device. По умолчанию: `gpu`
- `num_workers` - число воркеров. По умолчанию: `8`
- `iou_weight` - вес у IoU-компоненты функции потерь. По умолчанию: `1`
- `bce_weight` - вес у BCE-компоненты функции потерь. По умолчанию: `1`

2. **Передаём свои параметры**:

```bash
python src/train.py model=manet training.epochs=10 training.batch_size=10 optim.lr=2e-5
```

👆 Пример запуска команды, в которой запускаем обучение `manet` на `10` эпох c
размером батча равным 10 и learning rate равным `2e-5`

💡 Можно расширить список моделей, оптимизаторов, планировщиков путём создания
дополнительных конфиг-файлов. Также можно создавать собственные "комопзитные"
функции потерь (подобно IouBceLoss) с произвольным числом компонентов.

Графики обучения будут доступны по ссылке в `Weights and Biases`. Ссылка
появится в терминале.

Параметры эксперимента будут логироваться в папку `wandb/run-`, а веса лучшей
модели сохранятся в папке `logs/`

## ✅ Валидация

Оценим качество модели на val/test. В качестве CLI используется **Fire**:

1. **Стандартный вызов**:

```bash
python src/evaluate.py --weights=/path/to/your/weights.ckpt --model_type=your_model_type
```

Параметры:

- `weights` - путь до весов модели, обязательный аргумент
- `data` - путь до директории с данными. По умолчанию:
  `data/processed/gds_dataset`
- `subset` - тип выборки va/test. По умолчанию: `test`
- `batch_size` - размер батча. По умолчанию: `2`
- `num_workers` - число воркеров. По умолчанию: `4`
- `model_type` - тип модели. По умолчанию: `upernet`

⚠️ Лучше стараться указывать необходимый тип модели для корректного выбора
device для инференса. Некоторые модели не поддерживают инференс на `mps`
(macOs), поэтому в случае отсутствии доступа к `gpu` будут использовать `cpu`.

2. **Передаём свои параметры**:

```bash
python src/evaluate.py --weights=checkpoints/upernet.ckpt --subset=test --batch_size=5 --model_type=upernet
```

## 🔮 Инференс

1. **Стандартный вызов**

```bash
python src/inference.py --weights=/path/to/your/weights.ckpt --model_type=your_model_type
```

Параметры:

- `weights` - путь до весов модели, обязательный аргумент
- `inference_folder` - путь до директории с входными данными(изображения
  исходных топологий). По умолчанию:
  `data/processed/gds_dataset/origin/test_origin`
- `output_folder` - путь до директории, куда будут сохраняться изображений
  масочных коррекций. По умолчанию: `inference/output_img/`. Если директория
  отсутсвует, то она будет создана. Результаты будут непосредственно сохраняться
  в папок exp\_{i} внутри данной директории
- `batch_size` - размер батча. По умолчанию: `2`
- `num_workers` - число воркеров. По умолчанию: `4`
- `model_type` - тип модели. По умолчанию: `upernet`

⚠️ Лучше стараться указывать необходимый тип модели для корректного выбора
device для инференса. Некоторые модели не поддерживают инференс на `mps`
(macOs), поэтому в случае отсутствии доступа к `gpu` будут использовать `cpu`.

2. **Передаём свои параметры**:

```bash
python src/inference.py --weights=checkpoints/upernet.ckpt --model_type=upernet --batch_size=10
```

## ⚡ ONNX

Экспортируем веса модели в **onnx** формат

```bash
python python models/to_onnx.py --raw_weights=/path/to/your/weights.ckpt --onnx_weights=/path/to/save/onnx/weights.onnx
```

⚠️ На данный момент может возникнуть проблема при экспорте моделей библиотеки
segmentation_models_pytorch, так как в них есть слой адаптивного пуллинга,
который не очень дружит с onnx. Пока не успел исправить. В собственной
реализации `unet` такой проблемы не должно быть.

## Метрики

Качество генерации изображений оценивалось с использованием метрик **Pixel
Accuracy** и **IoU** (Intersection over Union). Дополнительно оценивалось
среднее время генерации одного изображения с масочной коррекцией.

## Графики

По мере работы в [этом](https://api.wandb.ai/links/ml_team_mskv/6kunkn1r) отчёте
будут появляться графики обучения.

## Литература

1. Zheng S. [и др.]. LithoBench: Benchmarking AI Computational Lithography for
   Semiconductor Manufacturing под ред. A. Oh [и др.]., Curran Associates, Inc.,
   2023.C. 30243–30254.
   [[cтатья](http://www.cse.cuhk.edu.hk/~byu/papers/C190-NeurIPS2023-LithoBench.pdf)];
   [[репозиторий](https://github.com/shelljane/lithobench)]
2. Chen G. [и др.]. DAMO: Deep Agile Mask Optimization for Full Chip Scale 2020.
   [[статья](https://www.cse.cuhk.edu.hk/~byu/papers/C104-ICCAD2020-DAMO.pdf)]
3. Yang H., Ren H. Enabling Scalable AI Computational Lithography with
   Physics-Inspired Models Tokyo Japan: ACM, 2023.C. 715–720.
   [[статья](https://d1qx31qr3h6wln.cloudfront.net/publications/Enabling_Scalable_AI_Computational_Lithography_with_Physics-Inspired_Models.pdf)]
