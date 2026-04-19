# Задачи с LSTM для прогнозирования временных рядов

## Задача 1: Базовое прогнозирование цен акций

LSTM модель для прогнозирования цены закрытия акций Apple (AAPL) на основе исторических данных.

**Файл:** [task1.py](https://github.com/kokodae/LSTM-tasks/blob/main/task1.py)

![Результат](https://github.com/kokodae/LSTM-tasks/blob/main/result1.png)

**Параметры модели:**
- Окно (window_size): 60 дней
- Обучающая выборка: 80% данных
- Слои LSTM: 2 слоя по 50 нейронов
- Полносвязные слои: 25 и 1 нейрон
- Оптимизатор: adam
- Функция потерь: mean_squared_error
- Эпохи: 5
- Batch size: 1

**Зависимости:**
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow

---

## Задача 2: Сравнение обычного и распределенного обучения

Расширенная версия с обучением на нескольких GPU (MirroredStrategy) и сравнением результатов.

**Файл:** [task2.py](https://github.com/kokodae/LSTM-tasks/blob/main/task2.py)

![Результат](https://github.com/kokodae/LSTM-tasks/blob/main/result2.png)
![РезультатСравнение](https://github.com/kokodae/LSTM-tasks/blob/main/result3.png)

**Особенности:**
- Сохранение моделей в файлы (lstm_model_normal.keras, lstm_model_distributed.keras)
- Сравнение loss и val_loss между обычной и распределенной моделью
- Визуализация прогнозов для обеих моделей
- Использование tf.distribute.MirroredStrategy для распределенного обучения

**Отличия от задачи 1:**
- Тестовая выборка формируется без перекрытия с обучающей
- Добавлена функция create_model() для повторного использования архитектуры
- Реализована функция predict_and_plot() для визуализации
- Добавлено сравнение графиков обучения

---

## Установка

pip install numpy pandas matplotlib yfinance scikit-learn tensorflow

## Примечания

1. Данные загружаются с Yahoo Finance (AAPL с 2010 по 2023)
2. Для распределенного обучения требуется несколько GPU или CPU
3. При отсутствии GPU распределенное обучение будет использовать CPU
