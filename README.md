
# 📊 Учебная практика
##  Построение и анализ ансамблевых моделей (Random Forest, Gradient Boosting): классификация

### 👥 Исполнители
- Студент 1: [Bairamov Alan]
- Студент 2: [Ermolaev Kirill]

### 📅 Срок выполнения
Июнь 2026 (6 недель)

## 🎯 Цель проекта
Построение и анализ ансамблевых моделей на основе решающих деревьев для задачи классификации: 
### датасет: Heart Disease Dataset
- **Источник:** [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv)
- **Автор:** johnsmith88 (оригинальные данные: Cleveland Clinic Foundation)
- **Лицензия:** Не указана (Unknown)
- **Задача:** Классификация — прогнозирование наличия заболевания сердца
- **Размер:** 303 объекта, 14 признаков


В рамках учебной практики выполняется построение и сравнительный анализ ансамблевых моделей машинного обучения:
- **Базовые модели:** Решающие деревья (Decision Trees)
- **Ансамблевые методы:** 
  - Bagging (Бэггинг)
  - Random Forest (Случайный лес)
  - Gradient Boosting (Градиентный бустинг: XGBoost, LightGBM, CatBoost, выбрать одну из моделей)


## 📁 Предварительная структура проекта
```
practice_DecisionTreeClassier/
│
├── data/                 # Данные (не загружаются в Git)
│   ├── raw/              # Исходные данные (не трогать!)
│   ├── processed/        # Очищенные, подготовленные данные
│   ├── README.md          # Описание датасета
│
├── notebooks/
│   ├── 01_EDA.ipynb      # Разведочный анализ
│   ├── 02_Baseline.ipynb # Базовые модели
│   ├── 03_RF.ipynb       # Random Forest
│   ├── 04_GB.ipynb       # Gradient Boosting
│   ├── 05_Comparison.ipynb # Сравнительный анализ
│   └── utils.py          # Вспомогательные функции
│
├── models/               # Сохраненные модели
│   ├── base_model.pkl
│   ├── rf_model.pkl
│   └── gb_model.pkl
│
── reports/
│   ├── figures/          # Графики и визуализации
│   ├── final_report.pdf  # Итоговый отчет
│
├── requirements.txt      # Зависимости
├── README.md             # Описание проекта
└── .gitignore            # Что игнорировать в Git

```

## 📥 Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/elena-kornaeva/Practice_DecisionTreeClassifier.git
cd Practice_DecisionTreeClassifier
```
или через GitHubDesktop (рекомендуется): File -> Clone Repository

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

или используя Anaconda Prompt (не обычный Command Prompt!):
```bash
conda install -r requirements.txt
```


## Загрузка данных
Скачайте датасеты с Kaggle и поместите их в предварительно созданную папку data/raw/:

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv

## Последовательность работы:
* 01_EDA.ipynb - Разведочный анализ данных и предобработка
* 02_Baseline_Models.ipynb - Обучение базовых моделей (Decision Tree)
* 03_Random_Forest.ipynb - Построение ансамблей (Bagging, Random Forest)
* 04_Gradient_Boosting.ipynb - Градиентный бустинг (XGBoost, LightGBM, CatBoost)
* 05_Comparison.ipynb - Сравнительный анализ и интерпретация (SHAP)

## 🔬 Методы и алгоритмы
Основные алгоритмы:
* Decision Trees: CART алгоритм с критериями:
* Классификация: Gini impurity, Entropy
* Bagging: Bootstrap aggregating для снижения variance
* Random Forest:
* Случайные подпространства признаков
* OOB (Out-of-Bag) валидация
* Feature Importance
* Gradient Boosting:
  - XGBoost: Градиентный бустинг с регуляризацией
  - LightGBM: Гистограммный бустинг
  - CatBoost: Обработка категориальных признаков

## Метрики качества:
Для классификации:
* Accuracy, Precision, Recall, F1-Score
* ROC-AUC, PR-AUC
* onfusion Matrix

## Дополнительные техники:
* Cross-Validation: StratifiedKFold / KFold
* Hyperparameter Tuning: RandomizedSearchCV, Optuna
* Feature Engineering: Создание новых признаков

## 📈 Ожидаемые результаты
Для классификации
* Метрики: ROC-AUC > 0.85, F1-Score > 0.80
* Интерпретация: Выявление ключевых факторов (например, курение для Stroke Prediction)
* Визуализация: ROC-кривые, Confusion Matrix

## 📚 Используемые библиотеки:
### Анализ данных:
* pandas >= 2.0.0
* numpy >= 1.24.0

### Машинное обучение:
* scikit-learn >= 1.3.0
* xgboost >= 2.0.0 
*  lightgbm >= 4.0.0
*  catboost >= 1.2.0

### Визуализация:*
* matplotlib >= 3.7.0
* seaborn >= 0.12.0

### Оптимизация: 
* optuna >= 3.0.0

### Jupyter:
* jupyter >= 1.0.0

## 📝 Рекомендации по работе

### Работа с Git:
* Делайте коммиты после каждого этапа
* Пишите понятные сообщения коммитов
* Синхронизируйтесь с напарником ежедневно

### Организация кода:
* Один ноутбук = один этап
* Используйте utils.py для общих функций
* Сохраняйте все графики в reports/figures/

### Эксперименты:
* Фиксируйте все гиперпараметры
* Ведите таблицу результатов (Excel/Google Sheets)
* Сохраняйте лучшие модели через joblib

### Отчетность:
* Начинайте писать отчет с первой недели
* Сохраняйте все важные графики
* Формулируйте выводы после каждого этапа

## 🎓 Образовательные цели
После завершения практики студенты смогут:

✅ Понимать математику ансамблевых методов

✅ Применять Bagging и Boosting на практике

✅ Проводить полный цикл ML-проекта

✅ Работать в команде с использованием Git

✅ Проводить вычислительные эксперименты


## 🎓 Инструкция по установке ПО:
* Скачайте Anaconda3 с https://www.anaconda.com/download/success (для windows)
* Установите с настройками по умолчанию (галочка "Register as default Python" — ДА, "Add to PATH" — НЕТ)
* Откройте Anaconda Prompt из меню Пуск
* Перейдите в папку проекта:

```bash
cd D:\...\Practice_DecisionTreeClassifier
```
* Доустановите библиотеки:

```bash
pip install xgboost catboost optuna
```

* Запустите Jupyter:
```bash
jupyter notebook
```
## Работайте:
1. Утром: открыть Anaconda Prompt
2. Перейти в папку проекта
```bash
cd D:\...\Practice_DecisionTreeClassifier
```

3. Запустить Jupyter
```bash
jupyter notebook
```

4. Работать в ноутбуке, сохранять (Ctrl+S)

5. Вечером: открыть GitHub Desktop
6. Сделать Commit to main + Push origin

### Полезные команды GitHub Desktop:

* Скачать изменения напарника: Fetch origin → Pull
* Загрузить свои изменения: Commit → Push origin
* Посмотреть историю: Вкладка History
* Открыть папку проекта: Repository → Show in Explorer