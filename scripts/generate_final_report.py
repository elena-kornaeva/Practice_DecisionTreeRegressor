from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from pathlib import Path

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

A4 = (8.27, 11.69)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 160,
    }
)


def money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def metric(value: float) -> str:
    return f"{value:.4f}"


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_DIR / "raw.csv")
    train = pd.read_csv(DATA_DIR / "processed" / "train_one_hot_encoded.csv")
    val = pd.read_csv(DATA_DIR / "processed" / "validation_one_hot_encoded.csv")
    test = pd.read_csv(DATA_DIR / "processed" / "test_one_hot_encoded.csv")
    return raw, train, val, test


def load_models() -> dict[str, object]:
    return {
        "Оптимизированное дерево": joblib.load(MODELS_DIR / "decision_tree_gridsearch_model.pkl"),
        "Bagging": joblib.load(MODELS_DIR / "bagging_gridsearch_model.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest_gridsearch_model.pkl"),
        "XGBoost": joblib.load(MODELS_DIR / "xgboost_randomized_search_model.pkl"),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["charges", "charges_original"])
    y = df["charges_original"]
    return x, y


def predict_original_scale(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(x))


def calculate_train_test_metrics(
    models: dict[str, object], train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)

    rows = []
    for model_name, model in models.items():
        for split_name, x, y in [
            ("Train", x_train, y_train),
            ("Test", x_test, y_test),
        ]:
            y_pred = predict_original_scale(model, x)
            rows.append(
                {
                    "Модель": model_name,
                    "Выборка": split_name,
                    "R2": r2_score(y, y_pred),
                    "MAE": mean_absolute_error(y, y_pred),
                    "RMSE": math.sqrt(mean_squared_error(y, y_pred)),
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORTS_DIR / "train_test_metrics_table.csv", index=False, encoding="utf-8")
    return metrics_df


def save_table_image(
    df: pd.DataFrame,
    path: Path,
    title: str,
    col_widths: list[float] | None = None,
    fontsize: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, max(2.0, 0.45 * len(df) + 1.2)))
    ax.axis("off")
    ax.set_title(title, pad=12, fontweight="bold")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#4d4d4d")
        if row == 0:
            cell.set_facecolor("#dbead2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f7fbf4" if row % 2 else "#ffffff")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_new_figures(raw: pd.DataFrame, train: pd.DataFrame, metrics_df: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    stats = raw["charges"].agg(["mean", "var", "std", "min", "max"]).rename(
        {
            "mean": "Среднее",
            "var": "Дисперсия",
            "std": "СКО",
            "min": "Минимум",
            "max": "Максимум",
        }
    )
    stats_df = pd.DataFrame(
        {
            "Статистика": stats.index,
            "Значение для charges": [money(v) for v in stats.values],
        }
    )
    paths["charges_stats"] = FIGURES_DIR / "charges_statistics_table.png"
    save_table_image(stats_df, paths["charges_stats"], "Таблица статистик целевой переменной (charges)")

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.boxplot(raw["charges"], vert=False, patch_artist=True, boxprops={"facecolor": "#9ecae1"})
    ax.set_xlabel("Стоимость страховки (charges), $")
    ax.set_yticks([1])
    ax.set_yticklabels(["Все клиенты"])
    ax.set_title("Boxplot целевой переменной (charges)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths["charges_boxplot_horizontal"] = FIGURES_DIR / "charges_boxplot_horizontal.png"
    fig.savefig(paths["charges_boxplot_horizontal"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = ["Некурящие", "Курящие"]
    values = [
        raw.loc[raw["smoker"] == "no", "charges"],
        raw.loc[raw["smoker"] == "yes", "charges"],
    ]
    bp = ax.boxplot(values, vert=False, patch_artist=True, tick_labels=labels)
    colors = ["#c7e9c0", "#fdae6b"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Стоимость страховки (charges), $")
    ax.set_title("Boxplot стоимости страховки по признаку курения (smoker)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths["charges_boxplot_by_smoker"] = FIGURES_DIR / "charges_boxplot_by_smoker.png"
    fig.savefig(paths["charges_boxplot_by_smoker"], bbox_inches="tight")
    plt.close(fig)

    numeric_cols = ["age", "bmi", "children"]
    corr_original = raw[numeric_cols + ["charges"]].corr()
    corr_log = raw[numeric_cols].copy()
    corr_log["log_charges"] = np.log1p(raw["charges"])
    corr_log = corr_log.corr()
    corr_matrices = [
        (corr_original, "До логарифмирования: corr(..., charges)"),
        (corr_log, "После логарифмирования: corr(..., log(charges + 1))"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, (corr_matrix, title) in zip(axes, corr_matrices):
        image = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=35, ha="right")
        ax.set_yticklabels(corr_matrix.index)
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cax = fig.add_axes([0.925, 0.24, 0.018, 0.50])
    fig.colorbar(image, cax=cax)
    fig.suptitle("Сравнение корреляций для обычного и логарифмированного таргета", fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.25, left=0.07, right=0.88, wspace=0.45)
    paths["correlation_original_vs_log"] = FIGURES_DIR / "correlation_matrix_original_vs_log_target.png"
    fig.savefig(paths["correlation_original_vs_log"], bbox_inches="tight")
    plt.close(fig)

    preview = train.head(6).copy()
    preview = preview[["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest", "charges"]]
    preview = preview.round(3)
    paths["processed_preview"] = FIGURES_DIR / "processed_dataset_preview.png"
    save_table_image(
        preview,
        paths["processed_preview"],
        "Фрагмент датасета после предобработки",
        fontsize=7,
    )

    table_df = metrics_df.copy()
    table_df["R2"] = table_df["R2"].map(metric)
    table_df["MAE"] = table_df["MAE"].map(money)
    table_df["RMSE"] = table_df["RMSE"].map(money)
    paths["train_test_metrics"] = FIGURES_DIR / "train_test_metrics_table.png"
    save_table_image(table_df, paths["train_test_metrics"], "Итоговые метрики на train и test", fontsize=8)

    xgb_results_path = REPORTS_DIR / "xgboost_results.json"
    if xgb_results_path.exists():
        xgb_results = json.loads(xgb_results_path.read_text(encoding="utf-8"))
        params = xgb_results["refined_best_params"]
        params_df = pd.DataFrame(
            {
                "Гиперпараметр": list(params.keys()),
                "Значение": [str(value) for value in params.values()],
            }
        )
        paths["xgboost_best_params"] = FIGURES_DIR / "xgboost_best_params.png"
        save_table_image(params_df, paths["xgboost_best_params"], "Лучшие параметры XGBoost", fontsize=8)

    r2_pivot = metrics_df.pivot(index="Модель", columns="Выборка", values="R2")
    order = ["Оптимизированное дерево", "Bagging", "Random Forest", "XGBoost"]
    r2_pivot = r2_pivot.loc[order]
    x = np.arange(len(r2_pivot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x - width / 2, r2_pivot["Train"], width, label="Train", color="#74a9cf")
    ax.bar(x + width / 2, r2_pivot["Test"], width, label="Test", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(r2_pivot.index, rotation=12, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Сравнение моделей по R2 на train и test")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["r2_train_test"] = FIGURES_DIR / "comparison_r2_train_test.png"
    fig.savefig(paths["r2_train_test"], bbox_inches="tight")
    plt.close(fig)

    return paths


def draw_wrapped(
    fig: plt.Figure,
    text: str,
    x: float,
    y: float,
    width: int = 92,
    size: float = 10.5,
    line_gap: float = 0.027,
    weight: str | None = None,
    family: str = "DejaVu Sans",
) -> float:
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph.strip() == "":
            y -= line_gap
            continue
        wrapped = textwrap.wrap(paragraph, width=width, replace_whitespace=False)
        for line in wrapped:
            fig.text(x, y, line, fontsize=size, fontfamily=family, fontweight=weight, va="top")
            y -= line_gap
        y -= line_gap * 0.35
    return y


def add_page_number(fig: plt.Figure, number: int) -> None:
    fig.text(0.5, 0.025, str(number), ha="center", va="bottom", fontsize=9)


def add_text_page(
    pdf: PdfPages,
    page_no: int,
    title: str,
    paragraphs: list[str],
    code: str | None = None,
    formulas: list[str] | None = None,
) -> int:
    fig = plt.figure(figsize=A4)
    y = 0.94
    fig.text(0.5, y, title, ha="center", va="top", fontsize=16, fontweight="bold")
    y -= 0.06
    for paragraph in paragraphs:
        y = draw_wrapped(fig, paragraph, 0.09, y)
        y -= 0.008
    if formulas:
        for formula in formulas:
            fig.text(0.5, y, formula, ha="center", va="top", fontsize=12)
            y -= 0.046
    if code:
        y -= 0.01
        fig.text(
            0.09,
            y,
            code,
            fontsize=8.2,
            fontfamily="DejaVu Sans Mono",
            va="top",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f2f2f2", "edgecolor": "#bdbdbd"},
        )
    add_page_number(fig, page_no)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def add_image_page(
    pdf: PdfPages,
    page_no: int,
    section_title: str,
    image_path: Path,
    caption: str,
    body: str | None = None,
) -> int:
    caption_title, caption_explanation = split_caption(caption)
    fig = plt.figure(figsize=A4)
    y = 0.95
    fig.text(0.5, y, section_title, ha="center", va="top", fontsize=15, fontweight="bold")
    y -= 0.055
    if body:
        y = draw_wrapped(fig, body, 0.09, y, width=90, size=10.2)
        y -= 0.015

    img = mpimg.imread(image_path)
    img_h, img_w = img.shape[:2]
    max_w, max_h = 0.82, 0.52
    aspect = img_w / img_h
    if aspect >= max_w / max_h:
        w = max_w
        h = w / aspect
    else:
        h = max_h
        w = h * aspect
    left = (1 - w) / 2
    bottom = max(0.24, y - h)
    ax = fig.add_axes([left, bottom, w, h])
    ax.imshow(img)
    ax.axis("off")

    caption_y = bottom - 0.035
    explanation_y = draw_wrapped(fig, caption_title, 0.09, caption_y, width=90, size=9.8)
    if caption_explanation:
        explanation_y -= 0.012
        draw_wrapped(fig, caption_explanation, 0.09, explanation_y, width=90, size=10.0)
    add_page_number(fig, page_no)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def split_caption(caption: str) -> tuple[str, str]:
    """Keep only the figure name in the caption; move interpretation below it."""
    title, sep, explanation = caption.partition(". ")
    if not sep:
        return caption.rstrip("."), ""
    return title.rstrip("."), explanation.strip()


def build_csv(rows: list[dict[str, str]]) -> None:
    path = REPORTS_DIR / "final_report.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Раздел", "Тип", "Заголовок", "Текст", "Изображение"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    raw, train, val, test = load_data()
    models = load_models()
    metrics_df = calculate_train_test_metrics(models, train, test)
    figure_paths = make_new_figures(raw, train, metrics_df)

    outliers_35000 = raw[raw["charges"] > 35000]
    outlier_count = len(outliers_35000)
    outlier_share = outlier_count / len(raw) * 100
    outlier_smoker_share = (outliers_35000["smoker"] == "yes").mean() * 100

    opt_tree = models["Оптимизированное дерево"]
    tree_depth = opt_tree.get_depth() if hasattr(opt_tree, "get_depth") else None
    tree_metrics = metrics_df[metrics_df["Модель"] == "Оптимизированное дерево"].set_index("Выборка")
    tree_train_r2 = tree_metrics.loc["Train", "R2"]
    tree_test_r2 = tree_metrics.loc["Test", "R2"]
    tree_train_rmse = tree_metrics.loc["Train", "RMSE"]
    tree_test_rmse = tree_metrics.loc["Test", "RMSE"]

    xgb_results_path = REPORTS_DIR / "xgboost_results.json"
    xgb_results = json.loads(xgb_results_path.read_text(encoding="utf-8")) if xgb_results_path.exists() else {}

    rows: list[dict[str, str]] = []

    def row(section: str, kind: str, title: str, text: str, image: str = "") -> None:
        rows.append({"Раздел": section, "Тип": kind, "Заголовок": title, "Текст": text, "Изображение": image})

    pdf_path = Path(os.environ.get("FINAL_REPORT_PDF", REPORTS_DIR / "final_report.pdf"))
    with PdfPages(pdf_path) as pdf:
        page = 1
        page = add_text_page(
            pdf,
            page,
            "Отчет по практике",
            [
                "Тема: прогнозирование стоимости медицинской страховки на основе дерева решений и ансамблевых моделей.",
                "В отчете рассматривается полный ход работы: постановка задачи, разведочный анализ данных, предобработка, построение базовой модели, борьба с переобучением, ансамблевые методы и итоговое сравнение моделей.",
                "Структура отчета оформлена по образцу: введение, основная часть, заключение, список использованных источников и приложения.",
            ],
        )
        page = add_text_page(
            pdf,
            page,
            "Содержание",
            [
                "Введение",
                "1. Постановка задачи и описание датасета",
                "2. Разведочный анализ и предобработка данных",
                "3. Базовая модель и проблема переобучения",
                "4. Ансамблевые модели: Bagging и Random Forest",
                "5. Градиентный бустинг",
                "6. Сравнительный анализ моделей",
                "Заключение",
                "Список использованных источников",
            ],
        )
        page = add_text_page(
            pdf,
            page,
            "Введение",
            [
                "Цель работы заключается в построении и сравнении моделей машинного обучения для прогнозирования стоимости медицинской страховки клиента. В качестве целевой переменной используется стоимость страховки (charges).",
                "Практическая значимость задачи состоит в том, что страховая стоимость зависит от нескольких факторов: возраста клиента (age), индекса массы тела (bmi), количества детей (children), пола (sex), факта курения (smoker) и региона проживания (region). Анализ этих признаков позволяет не только построить прогноз, но и понять, какие факторы сильнее всего влияют на итоговую стоимость.",
                "Для решения задачи применялись модели на основе решающих деревьев: оптимизированное дерево решений, Bagging, Random Forest и XGBoost. Качество оценивалось с помощью метрик R2, MAE и RMSE на тренировочной и тестовой выборках.",
            ],
        )
        row("Введение", "Текст", "Цель работы", "Построить и сравнить модели для прогнозирования стоимости страховки (charges).")

        page = add_text_page(
            pdf,
            page,
            "1. Постановка задачи и описание датасета",
            [
                "Решалась задача регрессии: по признакам клиента необходимо предсказать непрерывную величину стоимости медицинской страховки (charges). Исходный датасет содержит 1338 наблюдений и 7 столбцов.",
                "К количественным признакам относятся возраст клиента (age), индекс массы тела (bmi) и количество детей (children). К категориальным признакам относятся пол клиента (sex), факт курения (smoker) и регион проживания (region). Целевая переменная - стоимость страховки (charges).",
                "На первом этапе была выполнена проверка пропусков. Метод df.isna().sum() показал, что пропущенные значения отсутствуют, поэтому удаление строк и заполнение пропусков не потребовались.",
            ],
        )
        row("1. Постановка задачи и описание датасета", "Текст", "Описание данных", "Датасет содержит 1338 наблюдений, 6 признаков и целевую переменную (charges).")

        page = add_image_page(
            pdf,
            page,
            "1. Постановка задачи и описание датасета",
            figure_paths["charges_stats"],
            "Рисунок 1 - Статистики целевой переменной (charges). В таблице приведены среднее, дисперсия, среднеквадратическое отклонение, минимум и максимум. Большое отличие среднего значения от максимума показывает наличие дорогих страховых случаев.",
        )
        row("1. Постановка задачи и описание датасета", "Рисунок", "Статистики charges", "Добавлена таблица статистик целевой переменной (charges).", str(figure_paths["charges_stats"]))

        page = add_text_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            [
                "Для изучения целевой переменной было построено распределение стоимости страховки (charges). Распределение оказалось правосторонне асимметричным: основная масса наблюдений находится в нижнем диапазоне, но присутствует длинный правый хвост дорогих случаев.",
                f"Анализ выбросов показал, что {outlier_count} объектов, то есть {outlier_share:.1f}% выборки, имеют стоимость страховки выше $35 000. При детальном изучении выяснилось, что {outlier_smoker_share:.1f}% этих объектов являются курильщиками. Это позволяет считать такие наблюдения не ошибками данных, а реальными случаями высокой стоимости страхования для курящих клиентов.",
                "Для наглядности boxplot целевой переменной был представлен горизонтально, а распределение по признаку курения (smoker) показано отдельным boxplot. Такой вариант менее растянут по горизонтали и лучше показывает различия между курящими и некурящими клиентами.",
            ],
        )
        row("2. Разведочный анализ и предобработка данных", "Текст", "Анализ выбросов", f"Выше $35 000 находятся {outlier_count} объектов; {outlier_smoker_share:.1f}% из них - курильщики.")

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["charges_boxplot_horizontal"],
            "Рисунок 2 - Горизонтальный boxplot целевой переменной (charges). Видно, что в данных есть высокие значения, расположенные далеко правее основной массы наблюдений.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Boxplot charges", "Горизонтальный boxplot делает выбросы по (charges) более читаемыми.", str(figure_paths["charges_boxplot_horizontal"]))

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["charges_boxplot_by_smoker"],
            "Рисунок 3 - Boxplot стоимости страховки (charges) с разделением по признаку курения (smoker). У курящих клиентов медиана и верхние значения заметно выше, поэтому признак (smoker) является ключевым фактором стоимости.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Boxplot charges по smoker", "Показано, что высокие значения (charges) в основном связаны с курящими клиентами.", str(figure_paths["charges_boxplot_by_smoker"]))

        page = add_text_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            [
                "Так как целевая переменная (charges) имеет сильную правостороннюю асимметрию, для обучения моделей было применено логарифмирование. Модели обучались не на исходной стоимости, а на величине log(charges + 1). Это уменьшает влияние очень больших значений и делает обучение устойчивее.",
                "После получения прогноза выполнялось обратное преобразование. Поэтому все итоговые метрики в отчете рассчитаны не в логарифмической шкале, а в исходной шкале стоимости страховки.",
            ],
            formulas=[
                r"$y_{log}=\ln(y+1)$",
                r"$\hat{y}=\exp(\hat{y}_{log})-1$",
            ],
            code="df['charges_original'] = df['charges']\ndf['charges'] = np.log1p(df['charges'])\n\n# после прогноза модели\ny_pred = np.expm1(model.predict(X_test))",
        )
        row("2. Разведочный анализ и предобработка данных", "Формула", "Логарифмирование charges", "Для обучения использовано log1p(charges), а прогноз возвращался через expm1.")

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["correlation_original_vs_log"],
            "Рисунок 4 - Сравнение корреляционных матриц до и после логарифмирования целевой переменной. Сравнение показывает, как меняются парные линейные связи количественных признаков с таргетом. Для обычной стоимости страховки (charges) корреляция с возрастом (age) равна примерно 0.30, а после перехода к log(charges + 1) возраст связан с целевой переменной сильнее: около 0.53. Это показывает, что логарифмирование уменьшает влияние длинного правого хвоста и делает линейную зависимость более выраженной. Связь с индексом массы тела (bmi) после логарифмирования становится слабее, а связь с количеством детей (children) немного возрастает. Значимых сильных корреляций между самими количественными признаками не обнаружено.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Сравнение корреляционных матриц", "Добавлено сравнение корреляций с обычным таргетом (charges) и логарифмированным таргетом log(charges + 1).", str(figure_paths["correlation_original_vs_log"]))

        page = add_text_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            [
                "Категориальные признаки (sex), (smoker) и (region) были преобразованы с помощью OneHotEncoder. Такой способ кодирования не создает искусственный порядок между категориями и подходит для дальнейшего обучения моделей.",
                "С учетом перекоса выборки было принято решение использовать стратифицированное разделение данных. Для регрессии стандартная стратификация невозможна, потому что целевая переменная непрерывная. Поэтому была применена комбинированная стратификация: стоимость страховки (charges) сначала была разбита на интервалы, а затем интервалы были объединены с признаком курения (smoker). Это позволило сохранить похожее распределение стоимости и долю курильщиков в train, validation и test.",
                f"После разделения получены выборки: train - {len(train)} объект, validation - {len(val)} объектов, test - {len(test)} объектов. После OneHotEncoding итоговый набор признаков содержит {train.drop(columns=['charges', 'charges_original']).shape[1]} признаков.",
            ],
            code="df['charges_bin'] = pd.qcut(df['charges'], q=5, duplicates='drop')\ndf['stratify_col'] = df['charges_bin'].astype(str) + '_' + df['smoker']\n\ntrain_df, temp_df = train_test_split(\n    df,\n    test_size=0.4,\n    random_state=42,\n    stratify=df['stratify_col']\n)",
        )
        row("2. Разведочный анализ и предобработка данных", "Код", "Стратифицированное разделение", "Использована комбинированная стратификация через интервалы (charges) и признак (smoker).")

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["processed_preview"],
            "Рисунок 5 - Фрагмент датасета после предобработки. В таблице видны количественные признаки, one-hot признаки категорий и логарифмированная целевая переменная (charges). Исходная шкала сохранена отдельно в поле (charges_original) и используется для расчета метрик.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Итоговый датасет", "Показан фрагмент данных после OneHotEncoding и логарифмирования.", str(figure_paths["processed_preview"]))

        page = add_text_page(
            pdf,
            page,
            "Вывод по разделу 2",
            [
                "Таким образом, после выполненной предобработки был получен датасет без пропусков, с обработанными категориальными признаками и логарифмированной целевой переменной. Высокие значения стоимости страховки не удалялись, так как анализ показал их связь с признаком курения (smoker).",
                "Подготовленные данные подходят для обучения моделей регрессии: целевая переменная стабилизирована логарифмированием, признаки приведены к числовому виду, а train, validation и test имеют близкую структуру благодаря комбинированной стратификации.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            [
                "В качестве базовой модели было использовано дерево решений для регрессии. Сначала рассматривалась модель без сильных ограничений, чтобы показать риск переобучения: глубокое дерево способно запоминать отдельные наблюдения обучающей выборки и хуже переносить закономерности на новые данные.",
                f"Далее была построена оптимизированная модель дерева решений. Для нее итоговая глубина дерева составила {tree_depth}. На тренировочной выборке R2 = {tree_train_r2:.4f}, RMSE = {money(tree_train_rmse)}, а на тестовой выборке R2 = {tree_test_r2:.4f}, RMSE = {money(tree_test_rmse)}. Разница между train и test показывает, что ограничения дерева снизили переобучение, но полностью не устранили зависимость модели от структуры обучающей выборки.",
            ],
            formulas=[
                r"$R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$",
                r"$MAE=\frac{1}{n}\sum_i |y_i-\hat{y}_i|,\quad RMSE=\sqrt{\frac{1}{n}\sum_i (y_i-\hat{y}_i)^2}$",
            ],
        )
        row("3. Базовая модель и проблема переобучения", "Текст", "Оптимизированное дерево", f"Глубина дерева: {tree_depth}; train R2={tree_train_r2:.4f}; test R2={tree_test_r2:.4f}.")

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            REPORTS_DIR / "figures" / "decision_tree_best_params.png",
            "Рисунок 6 - Лучшие параметры оптимизированного дерева решений. В тексте используется именно термин 'оптимизированное дерево', чтобы не смешивать его с неограниченной базовой моделью.",
        )
        row("3. Базовая модель и проблема переобучения", "Рисунок", "Параметры дерева", "Лучшие параметры оптимизированного дерева решений.", "reports/figures/decision_tree_best_params.png")

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            REPORTS_DIR / "figures" / "decision_tree_residuals_test.png",
            "Рисунок 7 - Остатки оптимизированного дерева решений на тестовой выборке. Дискретность предсказаний связана с тем, что дерево возвращает одинаковое значение для всех объектов внутри одного листа. Крупные ошибки чаще возникают на дорогих страховых случаях.",
        )

        page = add_text_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            [
                "Для подбора гиперпараметров применялась процедура кросс-валидации. Ее смысл состоит в том, что обучающая выборка несколько раз делится на внутренние обучающие и проверочные части, а качество модели усредняется по разным разбиениям. Это снижает зависимость результата от одного случайного разделения.",
                "Для дерева изменялись параметры: максимальная глубина (max_depth), минимальное число объектов для разделения вершины (min_samples_split), минимальное число объектов в листе (min_samples_leaf), критерий разбиения (criterion) и параметр пост-обрезки (ccp_alpha).",
            ],
            code="param_grid = {\n    'max_depth': [4, 5, 6, 7, 8, 10, None],\n    'min_samples_split': [2, 5, 10, 20],\n    'min_samples_leaf': [1, 2, 4, 8],\n    'criterion': ['squared_error', 'absolute_error'],\n    'ccp_alpha': [0.0, 0.0001, 0.001]\n}\n\ngrid = GridSearchCV(\n    DecisionTreeRegressor(random_state=42),\n    param_grid=param_grid,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),\n    scoring=r2_original_scorer\n)",
        )

        page = add_text_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            [
                "После одиночного дерева были рассмотрены ансамблевые модели. Их смысл заключается в объединении нескольких деревьев, что снижает разброс предсказаний и делает модель устойчивее.",
                "Bagging обучает деревья на bootstrap-подвыборках и усредняет их прогнозы. Random Forest дополнительно использует случайный выбор признаков при построении разбиений, поэтому деревья внутри ансамбля становятся менее похожими друг на друга.",
                "Для Bagging подбирались число деревьев (n_estimators), доля объектов (max_samples), доля признаков (max_features), а также параметры базового дерева. Для Random Forest дополнительно подбирались параметры случайного леса: (max_depth), (min_samples_split), (min_samples_leaf), (max_features), (bootstrap) и (ccp_alpha).",
            ],
        )
        row("4. Ансамблевые модели", "Текст", "Bagging и Random Forest", "Ансамбли использовались для снижения разброса одиночного дерева.")

        page = add_image_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            REPORTS_DIR / "figures" / "bagging_best_params.png",
            "Рисунок 8 - Лучшие параметры Bagging. Модель подбиралась в два этапа: сначала широкий поиск, затем уточнение вокруг найденных значений.",
        )
        page = add_image_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            REPORTS_DIR / "figures" / "random_forest_best_params.png",
            "Рисунок 9 - Лучшие параметры Random Forest. По сравнению с Bagging здесь важен параметр (max_features), так как случайный лес выбирает подмножество признаков при разбиениях.",
        )

        page = add_text_page(
            pdf,
            page,
            "4. Ансамблевые модели: OOB-score",
            [
                "Для моделей с bootstrap-подвыборками дополнительно применялся OOB-score. Его смысл в том, что каждое дерево обучается не на всех объектах, и оставшиеся объекты можно использовать как внутреннюю проверочную выборку без отдельного разбиения.",
                "OOB-score сравнивался с оценкой по кросс-валидации. Близость этих значений показывает, что внутренняя оценка качества у ансамблей согласуется с KFold-кросс-валидацией. Для одиночного дерева OOB-score не применяется, потому что оно не строится на bootstrap-композиции.",
            ],
            code="bagging_oob = BaggingRegressor(\n    estimator=best_tree,\n    n_estimators=best_n,\n    bootstrap=True,\n    oob_score=True,\n    random_state=42\n)\n\nrf_oob = RandomForestRegressor(\n    n_estimators=best_n,\n    bootstrap=True,\n    oob_score=True,\n    random_state=42\n)",
        )

        page = add_text_page(
            pdf,
            page,
            "Вывод по разделу 4",
            [
                "Ансамбли улучшили качество по сравнению с оптимизированным одиночным деревом по R2 и RMSE. При этом значения RMSE у Bagging и Random Forest оказались близкими, потому что обе модели основаны на усреднении деревьев и работают с одним и тем же ограничением данных: высокие страховые случаи остаются наиболее сложной частью выборки.",
                "Random Forest показал небольшое преимущество за счет декорреляции деревьев через случайный выбор признаков. Однако разница с Bagging невелика, так как ключевые признаки, особенно факт курения (smoker), настолько сильны, что модели приходят к похожим разбиениям.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "5. Градиентный бустинг",
            [
                "Градиентный бустинг строит ансамбль последовательно: каждое новое дерево исправляет ошибки предыдущих деревьев. В соответствии с планом проекта для этого этапа использовалась библиотека XGBoost, а именно модель XGBRegressor.",
                "XGBoost является современной реализацией градиентного бустинга: кроме последовательного исправления ошибок, он использует регуляризацию, подвыборку объектов и подвыборку признаков, что помогает контролировать переобучение.",
                f"После широкого поиска лучший CV R2 составил {xgb_results.get('coarse_best_score', float('nan')):.4f}, после уточненного поиска - {xgb_results.get('refined_best_score', float('nan')):.4f}. Лучший шаг по validation был около {xgb_results.get('best_stage', 'NA')} деревьев.",
            ],
            code="xgb_base = XGBRegressor(\n    objective='reg:squarederror',\n    random_state=42,\n    n_jobs=-1,\n    verbosity=0\n)\n\nxgb_search = RandomizedSearchCV(\n    estimator=xgb_base,\n    param_distributions=xgb_param_grid,\n    n_iter=60,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),\n    scoring=r2_original_scorer,\n    random_state=42\n)",
        )
        row("5. Градиентный бустинг", "Текст", "Двухэтапный подбор", "Сначала широкий RandomizedSearchCV, затем уточнение вокруг лучших параметров.")

        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг",
            figure_paths["xgboost_best_params"],
            "Рисунок 10 - Лучшие параметры XGBoost после уточненного поиска. Параметры (learning_rate), (n_estimators), (max_depth), (subsample), (colsample_bytree), (reg_lambda) и (reg_alpha) управляют скоростью, сложностью и регуляризацией модели.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг",
            REPORTS_DIR / "figures" / "xgboost_learning_curve.png",
            "Рисунок 11 - Кривая качества XGBoost по числу деревьев. После определенного числа итераций качество на validation почти не растет, поэтому дальнейшее увеличение ансамбля может не давать практического выигрыша.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг",
            REPORTS_DIR / "figures" / "xgboost_residuals_test.png",
            "Рисунок 12 - Остатки XGBoost на тестовой выборке. Основная масса ошибок сосредоточена около нуля, но на высоких значениях (charges) сохраняется увеличенный разброс.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг",
            REPORTS_DIR / "figures" / "xgboost_feature_importance.png",
            "Рисунок 13 - Важность признаков XGBoost. Наиболее важным фактором является признак курения (smoker_yes), что подтверждает гипотезу из разведочного анализа о сильном влиянии курения на стоимость страховки.",
        )

        page = add_text_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            [
                "На заключительном этапе результаты всех моделей были объединены в одну таблицу. Метрики сравнивались на тренировочной и тестовой выборках, так как именно это позволяет оценить не только качество, но и степень переобучения.",
                "Особое внимание уделялось сравнению R2 на train и test. Если качество на train значительно выше, чем на test, модель переобучается. Ансамблевые модели показывают более стабильную разницу между train и test, чем одиночное дерево.",
            ],
        )

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            figure_paths["train_test_metrics"],
            "Рисунок 14 - Итоговая таблица метрик на train и test для всех моделей. Метрики рассчитаны в исходной шкале стоимости страховки после обратного преобразования expm1.",
        )
        row("6. Сравнительный анализ моделей", "Рисунок", "Train/test метрики", "Добавлена итоговая таблица метрик на train и test.", str(figure_paths["train_test_metrics"]))

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            figure_paths["r2_train_test"],
            "Рисунок 15 - Сравнение моделей по R2 на train и test. Столбцы разных цветов показывают, насколько качество на тестовой выборке отстает от тренировочной. Такой вид наглядно демонстрирует различия в переобучении одиночного оптимизированного дерева и ансамблей.",
        )
        row("6. Сравнительный анализ моделей", "Рисунок", "R2 train/test", "Добавлены разные цвета для train и test.", str(figure_paths["r2_train_test"]))

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_errors_models.png",
            "Рисунок 16 - Сравнение моделей по MAE и RMSE на тестовой выборке. RMSE у ансамблей близкие, потому что наибольший вклад в эту метрику дают редкие дорогие страховые случаи, сложные для всех моделей.",
        )
        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_feature_importance_all_models.png",
            "Рисунок 17 - Сравнение важности признаков. Во всех ансамблях главным фактором остается курение (smoker_yes), далее идут возраст (age) и индекс массы тела (bmi). Признаки региона и пола практически не влияют на целевую переменную.",
        )
        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_shap_summary_random_forest.png",
            "Рисунок 18 - SHAP-анализ Random Forest. SHAP-значения рассчитаны для модели, обученной на логарифмированной целевой переменной, поэтому они объясняют вклад признаков в прогноз log(charges + 1). График подтверждает высокую важность признаков (smoker_yes), (age) и (bmi).",
        )

        best_test = metrics_df[metrics_df["Выборка"] == "Test"].sort_values("R2", ascending=False).iloc[0]
        page = add_text_page(
            pdf,
            page,
            "Заключение",
            [
                f"В ходе работы была решена задача прогнозирования стоимости медицинской страховки (charges). Были выполнены разведочный анализ, обработка категориальных признаков, логарифмирование целевой переменной, стратифицированное разделение данных и обучение нескольких моделей.",
                f"Лучшее значение R2 на тестовой выборке показала модель {best_test['Модель']}: R2 = {best_test['R2']:.4f}, MAE = {money(best_test['MAE'])}, RMSE = {money(best_test['RMSE'])}. XGBoost улучшил результат Random Forest по R2 и RMSE, а Bagging также улучшил качество по сравнению с одиночным оптимизированным деревом по R2 и RMSE.",
                "Близкие значения RMSE у ансамблей объясняются тем, что все модели сталкиваются с одной и той же сложной областью данных - дорогими страховыми случаями. Эти наблюдения в основном связаны с курением, поэтому признак (smoker_yes) стабильно оказывается главным фактором прогноза. Признаки пола и региона в сравнении с ним имеют слабое влияние.",
            ],
        )
        row("Заключение", "Текст", "Итоговый вывод", f"Лучшая модель по test R2: {best_test['Модель']}.")

        page = add_text_page(
            pdf,
            page,
            "Список использованных источников",
            [
                "1. Scikit-learn documentation: DecisionTreeRegressor, BaggingRegressor, RandomForestRegressor, GridSearchCV, RandomizedSearchCV.",
                "2. XGBoost documentation: XGBRegressor and gradient boosted decision trees.",
                "3. Pandas documentation: работа с табличными данными и расчет описательных статистик.",
                "4. Matplotlib and Seaborn documentation: построение графиков, boxplot, heatmap и визуализация результатов.",
                "5. SHAP documentation: интерпретация моделей машинного обучения на основе SHAP-значений.",
                "6. Medical Cost Personal Datasets: набор данных о стоимости медицинской страховки.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "Приложение",
            [
                "В приложении приведены ключевые формулы и фрагменты кода, которые использовались в работе. Полная реализация находится в Jupyter Notebook-файлах проекта.",
            ],
            formulas=[
                r"$y_{log}=\ln(y+1)$",
                r"$\hat{y}=\exp(\hat{y}_{log})-1$",
                r"$RMSE=\sqrt{MSE}$",
            ],
        )

    build_csv(rows)
    print(f"PDF saved: {pdf_path}")
    print(f"CSV saved: {REPORTS_DIR / 'final_report.csv'}")


if __name__ == "__main__":
    main()
