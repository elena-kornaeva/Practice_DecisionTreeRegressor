from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree

from generate_final_report import (
    PROJECT_DIR,
    REPORTS_DIR,
    calculate_split_metrics,
    calculate_unrestricted_tree_metrics,
    ensure_dirs,
    load_data,
    load_models,
    make_new_figures,
    money,
    save_table_image,
)


HTML_PATH = REPORTS_DIR / "final_report.html"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = REPORTS_DIR / "assets"


TOC_ITEMS = [
    ("Введение", "Введение"),
    ("1. Постановка задачи и описание датасета", "1. Постановка задачи"),
    ("2. Разведочный анализ и предобработка данных", "2. Разведочный анализ"),
    ("3. Базовая модель и проблема переобучения", "3. Базовая модель"),
    ("4. Ансамблевые модели: Bagging и Random Forest", "4. Ансамблевые модели"),
    ("5. Градиентный бустинг: XGBoost", "5. Градиентный бустинг"),
    ("6. Сравнительный анализ моделей", "6. Сравнительный анализ"),
    ("Заключение", "Заключение"),
    ("Список использованных источников", "Список использованных источников"),
]


FIGURES = [
    ("raw_dataset_preview.png", "Фрагмент исходного датасета"),
    ("code_data_quality", "Проверка пропусков, дубликатов и удаление повторяющейся строки"),
    ("charges_distribution_report.png", "Распределение целевой переменной (charges)"),
    ("charges_boxplot_horizontal.png", "Диаграмма размаха целевой переменной (charges)"),
    ("charges_boxplot_by_smoker.png", "Стоимость страховки в группах по признаку курения (smoker)"),
    ("code_log_target", "Логарифмирование целевой переменной и обратное преобразование прогноза"),
    ("correlation_matrix_original_vs_log_target.png", "Корреляционные матрицы: а) с (charges); б) с log(charges + 1)"),
    ("stratified_split_code.png", "Код комбинированного стратифицированного разделения"),
    ("processed_dataset_preview.png", "Фрагмент датасета после предобработки"),
    ("decision_tree_structure_first_3_levels.png", "Первые три уровня оптимизированного дерева решений"),
    ("code_tree_grid", "Гиперпараметры широкого поиска оптимизированного дерева"),
    ("decision_tree_best_params.png", "Лучшие гиперпараметры оптимизированного дерева решений"),
    ("decision_tree_overfitting_metrics.png", "Метрики дерева до и после оптимизации"),
    ("decision_tree_residuals_test.png", "Остатки оптимизированного дерева на тестовой выборке"),
    ("code_bagging_grid", "Гиперпараметры широкого поиска модели Bagging"),
    ("bagging_best_params.png", "Лучшие гиперпараметры Bagging"),
    ("code_random_forest_grid", "Гиперпараметры широкого поиска модели Random Forest"),
    ("random_forest_best_params.png", "Лучшие гиперпараметры Random Forest"),
    ("code_oob_models", "Настройка OOB-оценки моделей Bagging и Random Forest"),
    ("oob_kfold_test_comparison.png", "Сравнение оценок OOB, KFold и test"),
    ("random_forest_feature_importance.png", "Важность признаков Random Forest"),
    ("code_xgboost_grid", "Гиперпараметры широкого поиска модели XGBoost"),
    ("xgboost_best_params.png", "Лучшие гиперпараметры XGBoost"),
    ("xgboost_learning_curve.png", "Обоснование выбора 648 деревьев для XGBoost"),
    ("xgboost_residuals_test.png", "Остатки XGBoost на тестовой выборке"),
    ("xgboost_feature_importance.png", "Важность признаков XGBoost"),
    ("comparison_r2_train_test.png", "Сравнение моделей по R2 на train и test"),
    ("comparison_errors_models.png", "Сравнение MAE и RMSE на тестовой выборке"),
    ("comparison_actual_vs_predicted_all_models.png", "Предсказанные и фактические значения для всех моделей"),
    ("comparison_residuals_all_models.png", "Остатки моделей на тестовой выборке"),
    ("comparison_feature_importance_all_models.png", "Сравнение важности признаков"),
    ("comparison_shap_summary_random_forest.png", "SHAP-анализ модели Random Forest"),
    ("comparison_inference_speed.png", "Среднее время инференса моделей"),
]


CSS = r"""
@page {
    size: A4;
    margin: 22mm 22mm 22mm 25mm;
    @bottom-center {
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 11pt;
    }
}

@page title {
    size: A4;
    margin: 20mm 15mm 20mm 30mm;
    @bottom-center { content: ""; }
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    color: #000;
    background: #fff;
    font-family: "Times New Roman", Times, serif;
    font-size: 14pt;
    line-height: 1.5;
}

body { counter-reset: figure table formula; }

.title-page {
    page: title;
    height: 257mm;
    break-after: page;
    position: relative;
    text-align: center;
}

.university-block {
    font-size: 14pt;
    line-height: 1.25;
    font-weight: bold;
    text-transform: uppercase;
}

.report-kind {
    margin-top: 66mm;
    font-size: 14pt;
    font-weight: bold;
    text-transform: uppercase;
    line-height: 1.35;
}

.title-label {
    margin-top: 26mm;
    font-size: 14pt;
    font-weight: bold;
}

.report-title {
    margin: 1mm auto 0;
    max-width: 165mm;
    font-size: 14pt;
    font-weight: bold;
    line-height: 1.3;
    text-transform: uppercase;
}

.people {
    position: absolute;
    top: 176mm;
    right: 0;
    width: 108mm;
    text-align: left;
    font-size: 14pt;
    line-height: 1.25;
}

.people-group {
    display: grid;
    grid-template-columns: 38mm 1fr;
    column-gap: 3mm;
}
.people-group + .people-group { margin-top: 10mm; }
.people-label { font-weight: bold; }

.title-bottom {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    line-height: 1.4;
}

.front-page { break-after: page; }
.chapter { break-before: page; }

h1 {
    margin: 0 0 10mm;
    font-size: 18pt;
    line-height: 1.25;
    text-align: center;
    font-weight: bold;
}

h2 {
    margin: 8mm 0 4mm;
    font-size: 16pt;
    line-height: 1.3;
    text-align: left;
    font-weight: bold;
    break-after: avoid;
}

h3 {
    margin: 6mm 0 3mm;
    font-size: 14pt;
    line-height: 1.3;
    text-align: left;
    font-weight: bold;
    break-after: avoid;
}

p {
    margin: 0 0 2.5mm;
    text-align: justify;
    text-indent: 12.5mm;
    orphans: 3;
    widows: 3;
}

.no-indent { text-indent: 0; }
.center { text-align: center; text-indent: 0; }

.toc-list,
.figure-list,
.source-list {
    margin: 0;
    padding: 0;
    list-style: none;
}

.toc-list li {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 2mm;
    margin: 0 0 3mm;
    line-height: 1.3;
}

.toc-dots { border-bottom: 1px dotted #555; transform: translateY(-2mm); }

.figure-list li {
    margin: 0 0 1.2mm;
    padding-left: 10mm;
    text-indent: -10mm;
    font-size: 12pt;
    line-height: 1.2;
}

.figure-block {
    margin: 5mm 0 3mm;
    break-inside: auto;
}

.compact-figure { margin: 3mm 0 2mm; }
.code-figure-block { margin: 1.5mm 0 1.5mm; }

figure {
    margin: 0;
    text-align: center;
    counter-increment: figure;
    break-inside: avoid;
}

figure img {
    display: block;
    max-width: 100%;
    max-height: 105mm;
    width: auto;
    height: auto;
    object-fit: contain;
    margin: 0 auto 3mm;
}

figure.tree-figure img { max-height: 92mm; }
figure.wide-figure img { max-height: 90mm; }
figure.correlation-figure img { max-height: 142mm; }
figure.code-figure img { width: 88%; max-height: 70mm; }
figure.dataset-figure img { max-height: 48mm; }
figure.code-snippet pre {
    margin: 0 auto 3mm;
    width: 94%;
    text-align: left;
}

figcaption {
    font-size: 12pt;
    line-height: 1.25;
    text-align: center;
}

.figure-explanation {
    margin-top: 3mm;
    text-indent: 12.5mm;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 5mm 0;
    font-size: 11pt;
    line-height: 1.25;
    break-inside: avoid;
    counter-increment: table;
}

caption {
    caption-side: top;
    margin-bottom: 2mm;
    text-align: center;
    font-size: 12pt;
    font-weight: normal;
}

th, td {
    border: 1px solid #333;
    padding: 1.6mm 1.8mm;
    text-align: center;
    vertical-align: middle;
}

th { font-weight: bold; }
.metrics-table { font-size: 9.3pt; }
.metrics-table th, .metrics-table td { padding: 1.1mm 1mm; }
.compact-table { margin: 2.5mm 0; }
.compact-table th, .compact-table td { padding: 1.1mm 1.4mm; }

.formula {
    position: relative;
    margin: 4mm 0;
    padding: 0 18mm;
    text-align: center;
    font-size: 14pt;
    line-height: 1.5;
    break-inside: avoid;
    counter-increment: formula;
}

.formula::after {
    content: "(" counter(formula) ")";
    position: absolute;
    top: 0;
    right: 0;
    font-size: 14pt;
    font-style: normal;
}

pre {
    margin: 4mm 0 5mm;
    padding: 3mm 4mm;
    border: 1px solid #777;
    background: #f5f5f5;
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    line-height: 1.25;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    break-inside: avoid;
}

.source-list { counter-reset: source; }
.source-list li {
    counter-increment: source;
    margin: 0 0 3mm;
    padding-left: 10mm;
    text-indent: -10mm;
    text-align: justify;
}
.source-list li::before { content: counter(source) ". "; }

a { color: #000; text-decoration: none; }
.citation { white-space: nowrap; }
.source-list a {
    text-decoration: underline;
    overflow-wrap: anywhere;
}
"""


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def paragraph(text: str, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<p{class_attr}>{e(text)}</p>"


def paragraph_html(text: str, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<p{class_attr}>{text}</p>"


def citation(*numbers: int) -> str:
    links = ", ".join(
        f'<a class="citation" href="#source-{number}">[{number}]</a>'
        for number in numbers
    )
    return links


def code_figure(code: str, number: int, title: str, explanation: str = "") -> str:
    explanation_html = (
        f'<p class="figure-explanation">{e(explanation)}</p>' if explanation else ""
    )
    return (
        '<div class="figure-block code-figure-block">'
        '<figure class="code-snippet">'
        f'<pre>{e(code)}</pre>'
        f'<figcaption>Рисунок {number} - {e(title)}</figcaption>'
        '</figure>'
        f'{explanation_html}'
        '</div>'
    )


def figure(filename: str, number: int, title: str, explanation: str, css_class: str = "") -> str:
    path = FIGURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден рисунок: {path}")
    class_attr = f" {css_class}" if css_class else ""
    return (
        '<div class="figure-block">'
        f'<figure class="{class_attr.strip()}">'
        f'<img src="figures/{e(filename)}" alt="{e(title)}">'
        f"<figcaption>Рисунок {number} - {e(title)}</figcaption>"
        "</figure>"
        f'<p class="figure-explanation">{e(explanation)}</p>'
        "</div>"
    )


def labeled_figure(
    filename: str,
    label: str,
    title: str,
    explanation: str,
    css_class: str = "correlation-figure",
    compact: bool = False,
) -> str:
    path = FIGURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден рисунок: {path}")
    block_class = "figure-block compact-figure" if compact else "figure-block"
    explanation_html = (
        f'<p class="figure-explanation">{e(explanation)}</p>' if explanation else ""
    )
    return (
        f'<div class="{block_class}">'
        f'<figure class="{e(css_class)}">'
        f'<img src="figures/{e(filename)}" alt="{e(title)}">'
        f"<figcaption>Рисунок {e(label)} - {e(title)}</figcaption>"
        "</figure>"
        f"{explanation_html}"
        "</div>"
    )


def make_stratification_code_figure() -> None:
    code = (
        "df['charges_bin'] = pd.qcut(df['charges'], q=5, duplicates='drop')\n"
        "df['strata'] = df['charges_bin'].astype(str) + '_' + df['smoker']\n\n"
        "train_df, temp_df = train_test_split(\n"
        "    df, test_size=0.40, random_state=42, stratify=df['strata']\n"
        ")\n"
        "validation_df, test_df = train_test_split(\n"
        "    temp_df, test_size=0.50, random_state=42, stratify=temp_df['strata']\n"
        ")"
    )
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        code,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=9.5,
        linespacing=1.28,
        bbox={
            "boxstyle": "square,pad=0.75",
            "facecolor": "#f7f7f7",
            "edgecolor": "#666666",
            "linewidth": 0.9,
        },
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        FIGURES_DIR / "stratified_split_code.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def make_report_specific_figures(
    models: dict[str, object],
    metrics_df: pd.DataFrame,
    unrestricted_metrics: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    feature_columns = [
        column for column in test.columns if column not in ["charges", "charges_original"]
    ]

    tree_figure, tree_axis = plt.subplots(figsize=(15.5, 7.1))
    plot_tree(
        models["Оптимизированное дерево"],
        max_depth=3,
        feature_names=feature_columns,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=tree_axis,
    )
    tree_figure.tight_layout(pad=0.3)
    tree_figure.savefig(
        FIGURES_DIR / "decision_tree_structure_first_3_levels.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(tree_figure)

    parameter_specs = {
        "decision_tree_best_params.png": (
            models["Оптимизированное дерево"],
            ["criterion", "max_depth", "min_samples_split", "min_samples_leaf", "ccp_alpha"],
        ),
        "bagging_best_params.png": (
            models["Bagging"],
            [
                "n_estimators",
                "max_samples",
                "max_features",
                "estimator__max_depth",
                "estimator__min_samples_split",
                "estimator__min_samples_leaf",
            ],
        ),
        "random_forest_best_params.png": (
            models["Random Forest"],
            [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "bootstrap",
                "ccp_alpha",
            ],
        ),
        "xgboost_best_params.png": (
            models["XGBoost"],
            [
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ],
        ),
    }
    for filename, (model, parameter_names) in parameter_specs.items():
        parameters = model.get_params()
        parameter_table = pd.DataFrame(
            {
                "Гиперпараметр": parameter_names,
                "Значение": [str(parameters[name]) for name in parameter_names],
            }
        )
        save_table_image(
            parameter_table,
            FIGURES_DIR / filename,
            "",
            col_widths=[0.52, 0.38],
            fontsize=8,
            table_scale_y=1.25,
        )

    tree_comparison = pd.concat(
        [
            unrestricted_metrics,
            metrics_df[metrics_df["Модель"] == "Оптимизированное дерево"],
        ],
        ignore_index=True,
    )[["Модель", "Выборка", "R2", "MAE", "RMSE"]]
    tree_comparison["R2"] = tree_comparison["R2"].map(lambda value: f"{value:.4f}")
    tree_comparison["MAE"] = tree_comparison["MAE"].map(money)
    tree_comparison["RMSE"] = tree_comparison["RMSE"].map(money)
    save_table_image(
        tree_comparison,
        FIGURES_DIR / "decision_tree_overfitting_metrics.png",
        "",
        col_widths=[0.28, 0.15, 0.10, 0.15, 0.15],
        fontsize=8,
        table_scale_y=1.25,
    )

    oob_table = pd.DataFrame(
        [
            ["Оптимизированное дерево", "-", "0.8319", "0.8399"],
            ["Bagging", "0.8628", "0.8478", "0.8475"],
            ["Random Forest", "0.8637", "0.8466", "0.8485"],
        ],
        columns=["Модель", "OOB R2", "KFold R2", "Test R2"],
    )
    save_table_image(
        oob_table,
        FIGURES_DIR / "oob_kfold_test_comparison.png",
        "",
        col_widths=[0.30, 0.15, 0.15, 0.15],
        fontsize=8,
        table_scale_y=1.25,
    )

    test_metrics = metrics_df[metrics_df["Выборка"] == "Test"].set_index("Модель")
    model_order = ["XGBoost", "Random Forest", "Bagging", "Оптимизированное дерево"]
    labels = ["XGBoost", "Random Forest", "Bagging", "Оптимизированное\nдерево"]
    x = np.arange(len(model_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    mae_bars = ax.bar(
        x - width / 2,
        test_metrics.loc[model_order, "MAE"],
        width,
        label="MAE",
        color="#377eb8",
    )
    rmse_bars = ax.bar(
        x + width / 2,
        test_metrics.loc[model_order, "RMSE"],
        width,
        label="RMSE",
        color="#e6862a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Ошибка в исходной шкале charges")
    ax.set_ylim(0, float(test_metrics.loc[model_order, "RMSE"].max()) * 1.22)
    ax.margins(x=0.07)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncols=2)
    ax.bar_label(mae_bars, fmt="%.0f", padding=3, fontsize=8)
    ax.bar_label(rmse_bars, fmt="%.0f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_errors_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    actual = test["charges_original"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0))
    for ax, model_name in zip(
        axes.flat,
        ["Оптимизированное дерево", "Bagging", "Random Forest", "XGBoost"],
    ):
        predicted = np.expm1(models[model_name].predict(test[feature_columns]))
        lower = min(float(actual.min()), float(predicted.min()))
        upper = max(float(actual.max()), float(predicted.max()))
        ax.scatter(actual, predicted, alpha=0.58, color="#4c9acb", s=24)
        ax.plot([lower, upper], [lower, upper], color="#d62728", linestyle="--", linewidth=1.4)
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Фактические charges")
        ax.set_ylabel("Предсказанные charges")
        ax.grid(True, alpha=0.22)
    fig.tight_layout(h_pad=1.7, w_pad=1.4)
    fig.savefig(
        FIGURES_DIR / "comparison_actual_vs_predicted_all_models.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def table_html(headers: list[str], rows: list[list[str]], caption_text: str, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    head = "".join(f"<th>{e(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{e(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f"<table{class_attr}>"
        f"<caption>{e(caption_text)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def make_correlation_subfigures(raw: pd.DataFrame) -> None:
    numeric_cols = ["age", "bmi", "children"]
    original = raw[numeric_cols + ["charges"]].corr()
    logarithmic = raw[numeric_cols].copy()
    logarithmic["log_charges"] = np.log1p(raw["charges"])
    logarithmic = logarithmic.corr()

    configurations = [
        (
            original,
            "Корреляции с исходной целевой переменной (charges)",
            FIGURES_DIR / "correlation_matrix_original_target_large.png",
        ),
        (
            logarithmic,
            "Корреляции с логарифмом целевой переменной log(charges + 1)",
            FIGURES_DIR / "correlation_matrix_log_target_large.png",
        ),
    ]

    for matrix, title, path in configurations:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        image = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=25, ha="right", fontsize=10)
        ax.set_yticklabels(matrix.index, fontsize=10)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                text_color = "white" if abs(value) > 0.55 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color,
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def make_xgboost_learning_curve(
    model: object,
    train: pd.DataFrame,
    val: pd.DataFrame,
    reported_best_stage: int,
) -> dict[int, float]:
    x_train = train.drop(columns=["charges", "charges_original"])
    x_val = val.drop(columns=["charges", "charges_original"])
    y_train = train["charges"]
    y_val = val["charges"]
    curve_model = clone(model)
    curve_model.fit(x_train, y_train)
    n_estimators = int(curve_model.get_params()["n_estimators"])

    train_scores: list[float] = []
    validation_scores: list[float] = []
    for stage in range(1, n_estimators + 1):
        train_prediction = curve_model.predict(x_train, iteration_range=(0, stage))
        validation_prediction = curve_model.predict(x_val, iteration_range=(0, stage))
        train_scores.append(r2_score(np.expm1(y_train), np.expm1(train_prediction)))
        validation_scores.append(r2_score(np.expm1(y_val), np.expm1(validation_prediction)))

    calculated_best_stage = int(np.argmax(validation_scores) + 1)
    best_stage = reported_best_stage if reported_best_stage == calculated_best_stage else calculated_best_stage
    stages = np.arange(1, n_estimators + 1)
    checkpoints = sorted({400, 500, 600, best_stage, n_estimators})
    checkpoint_scores = {
        stage: validation_scores[stage - 1]
        for stage in checkpoints
        if 1 <= stage <= n_estimators
    }

    fig, (full_ax, zoom_ax) = plt.subplots(
        2,
        1,
        figsize=(9.4, 8.0),
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    full_ax.plot(stages, train_scores, label="Train R2", color="#377eb8", linewidth=1.8)
    full_ax.plot(stages, validation_scores, label="Validation R2", color="#4daf4a", linewidth=1.8)
    full_ax.axvline(best_stage, color="#d62728", linestyle="--", linewidth=1.5)
    full_ax.scatter(
        [best_stage],
        [validation_scores[best_stage - 1]],
        color="#d62728",
        marker="*",
        s=130,
        zorder=5,
        label=f"Максимум validation: {best_stage}",
    )
    full_ax.set_title("Качество XGBoost на всех шагах обучения", fontweight="bold")
    full_ax.set_xlabel("Количество деревьев")
    full_ax.set_ylabel("R2")
    full_ax.grid(True, alpha=0.25)
    full_ax.legend(loc="lower right")

    zoom_start = min(350, n_estimators)
    zoom_mask = stages >= zoom_start
    zoom_stages = stages[zoom_mask]
    zoom_scores = np.asarray(validation_scores)[zoom_mask]
    zoom_ax.plot(zoom_stages, zoom_scores, color="#4daf4a", linewidth=2.0)
    marker_offsets = {
        400: (0, -32),
        500: (0, -32),
        600: (-26, 24),
        best_stage: (0, -34),
        n_estimators: (0, 24),
    }
    for stage, score in checkpoint_scores.items():
        is_best = stage == best_stage
        zoom_ax.scatter(
            stage,
            score,
            color="#d62728" if is_best else "#2c7fb8",
            marker="*" if is_best else "o",
            s=125 if is_best else 55,
            zorder=5,
        )
        offset = marker_offsets.get(stage, (0, 14))
        zoom_ax.annotate(
            f"{stage}\nR2={score:.4f}",
            xy=(stage, score),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold" if is_best else "normal",
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8},
        )
    zoom_ax.axvline(best_stage, color="#d62728", linestyle="--", linewidth=1.4)
    zoom_ax.set_xlim(zoom_start, n_estimators + 10)
    zoom_padding = max((zoom_scores.max() - zoom_scores.min()) * 0.22, 0.001)
    zoom_ax.set_ylim(zoom_scores.min() - zoom_padding, zoom_scores.max() + zoom_padding)
    zoom_ax.set_xticks(list(checkpoint_scores))
    zoom_ax.set_title(
        f"Уточнение: максимум Validation R2 достигается на {best_stage} деревьях",
        fontweight="bold",
        pad=14,
    )
    zoom_ax.set_xlabel("Контрольные значения количества деревьев")
    zoom_ax.set_ylabel("Validation R2")
    zoom_ax.grid(True, alpha=0.25)

    fig.tight_layout(h_pad=3.2)
    fig.savefig(FIGURES_DIR / "xgboost_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return checkpoint_scores


def detect_page_map(pdf_path: Path | None) -> dict[str, int]:
    if pdf_path is None or not pdf_path.exists():
        return {}
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}

    page_map: dict[str, int] = {}
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        if page_number <= 4:
            continue
        text = page.extract_text() or ""
        for _label, marker in TOC_ITEMS:
            if marker in text and marker not in page_map:
                page_map[marker] = page_number
    return page_map


def build_report(page_map: dict[str, int]) -> str:
    ensure_dirs()
    raw_with_duplicates, train, val, test = load_data()
    duplicate_count = int(raw_with_duplicates.duplicated().sum())
    raw = raw_with_duplicates.drop_duplicates().reset_index(drop=True)
    models = load_models()
    metrics_df = calculate_split_metrics(models, train, val, test)
    unrestricted_tree, unrestricted_metrics = calculate_unrestricted_tree_metrics(train, val, test)
    make_new_figures(raw, train, metrics_df, unrestricted_metrics)
    make_report_specific_figures(models, metrics_df, unrestricted_metrics, test)
    make_correlation_subfigures(raw)
    make_stratification_code_figure()

    target_skew = float(raw["charges"].skew())
    log_target_skew = float(np.log1p(raw["charges"]).skew())
    mode_grid = np.linspace(float(raw["charges"].min()), float(raw["charges"].max()), 20_000)
    target_mode = float(mode_grid[np.argmax(gaussian_kde(raw["charges"])(mode_grid))])
    outlier_mask = raw["charges"] > 35000
    outlier_count = int(outlier_mask.sum())
    outlier_share = outlier_count / len(raw) * 100
    outlier_smoker_share = float((raw.loc[outlier_mask, "smoker"] == "yes").mean() * 100)

    best_tree = models["Оптимизированное дерево"]
    tree_depth = best_tree.get_depth()
    unrestricted_by_split = unrestricted_metrics.set_index("Выборка")
    unrestricted_train = unrestricted_by_split.loc["Train"]
    unrestricted_test = unrestricted_by_split.loc["Test"]
    metric_by_model_split = metrics_df.set_index(["Модель", "Выборка"])
    tree_train = metric_by_model_split.loc[("Оптимизированное дерево", "Train")]
    tree_val = metric_by_model_split.loc[("Оптимизированное дерево", "Validation")]
    tree_test = metric_by_model_split.loc[("Оптимизированное дерево", "Test")]
    bagging_test = metric_by_model_split.loc[("Bagging", "Test")]
    forest_test = metric_by_model_split.loc[("Random Forest", "Test")]
    xgb_train = metric_by_model_split.loc[("XGBoost", "Train")]
    xgb_val = metric_by_model_split.loc[("XGBoost", "Validation")]
    xgb_test = metric_by_model_split.loc[("XGBoost", "Test")]

    test_feature_columns = [
        column for column in test.columns if column not in ["charges", "charges_original"]
    ]
    test_actual = test["charges_original"].to_numpy()

    def residual_zone_statistics(model: object) -> dict[str, float | int]:
        prediction = np.expm1(model.predict(test[test_feature_columns]))
        residuals = test_actual - prediction
        inexpensive = test_actual <= 20_000
        expensive = ~inexpensive
        return {
            "inexpensive_n": int(inexpensive.sum()),
            "expensive_n": int(expensive.sum()),
            "inexpensive_std": float(np.std(residuals[inexpensive], ddof=1)),
            "expensive_std": float(np.std(residuals[expensive], ddof=1)),
            "inexpensive_mae": float(np.mean(np.abs(residuals[inexpensive]))),
            "expensive_mae": float(np.mean(np.abs(residuals[expensive]))),
        }

    tree_residual_zones = residual_zone_statistics(models["Оптимизированное дерево"])
    xgb_residual_zones = residual_zone_statistics(models["XGBoost"])

    xgb_results_path = REPORTS_DIR / "xgboost_results.json"
    xgb_results = json.loads(xgb_results_path.read_text(encoding="utf-8")) if xgb_results_path.exists() else {}
    reported_best_stage = int(xgb_results.get("best_stage", models["XGBoost"].get_params()["n_estimators"]))
    learning_curve_scores = make_xgboost_learning_curve(
        models["XGBoost"],
        train,
        val,
        reported_best_stage,
    )
    selected_learning_stage = max(learning_curve_scores, key=learning_curve_scores.get)

    figure_numbers = {filename: index for index, (filename, _title) in enumerate(FIGURES, start=1)}
    figure_titles = dict(FIGURES)

    def fig(filename: str, explanation: str, css_class: str = "") -> str:
        return figure(
            filename,
            figure_numbers[filename],
            figure_titles[filename],
            explanation,
            css_class,
        )

    def code_fig(identifier: str, code: str, explanation: str = "") -> str:
        return code_figure(
            code,
            figure_numbers[identifier],
            figure_titles[identifier],
            explanation,
        )

    parts: list[str] = []
    parts.append("<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">")
    parts.append("<title>Отчет по практике</title><style>")
    parts.append(CSS)
    parts.append("</style></head><body>")

    parts.append(
        """
        <section class="title-page">
          <div class="university-block">
            Автономная некоммерческая организация высшего<br>
            образования «Университет Иннополис»<br>
            (АНО ВО «Университет Иннополис»)
          </div>
          <div class="report-kind">Отчет<br>по учебной практике</div>
          <div class="title-label">на тему:</div>
          <div class="report-title">Построение и анализ ансамблевых моделей на основе<br>
          решающих деревьев: регрессия</div>
          <div class="people">
            <div class="people-group"><span class="people-label">Выполнили:</span><span>Камалов Булат<br>Умаханов Мансур</span></div>
            <div class="people-group"><span class="people-label">Руководитель:</span><span>к.ф.-м.н, доцент<br>Корнаева Е.П.</span></div>
          </div>
          <div class="title-bottom">Иннополис - 2026</div>
        </section>
        """
    )

    parts.append('<section class="front-page"><h1>Содержание</h1><ol class="toc-list">')
    for label, marker in TOC_ITEMS:
        page_number = page_map.get(marker, "")
        parts.append(
            f"<li><span>{e(label)}</span><span class=\"toc-dots\"></span><span>{e(page_number)}</span></li>"
        )
    parts.append("</ol></section>")

    parts.append('<section class="front-page"><h1>Список рисунков</h1><ol class="figure-list">')
    for number, (_filename, title) in enumerate(FIGURES, start=1):
        parts.append(f"<li>Рисунок {number} - {e(title)}</li>")
    parts.append("</ol></section>")

    parts.append('<section class="front-page"><h1>Список таблиц</h1><ol class="figure-list">')
    for number, title in enumerate(
        [
            "Описательные статистики целевой переменной (charges)",
            "Размеры итоговых выборок",
            "Метрики моделей на train, validation и test",
        ],
        start=1,
    ):
        parts.append(f"<li>Таблица {number} - {e(title)}</li>")
    parts.append("</ol></section>")

    parts.append('<section class="chapter" id="introduction"><h1>Введение</h1>')
    parts.append(paragraph_html(
        "Актуальность работы обусловлена ростом объема данных в страховании и необходимостью точнее оценивать индивидуальный риск клиента. Методы машинного обучения позволяют учитывать совместное влияние демографических и поведенческих факторов, благодаря чему расчет страховой стоимости становится более обоснованным. "
        + citation(1)
    ))
    parts.append(paragraph(
        "Цель работы - построить и сопоставить модели регрессии для прогнозирования стоимости медицинской страховки (charges), а также определить признаки, которые сильнее всего влияют на результат. В качестве факторов рассматриваются возраст клиента (age), индекс массы тела (bmi), количество детей (children), пол (sex), факт курения (smoker) и регион проживания (region)."
    ))
    parts.append(paragraph(
        "Исследование проводилось последовательно. После разведочного анализа и предобработки данных было построено одиночное дерево решений, затем ансамбли Bagging и Random Forest, а на заключительном этапе - модель XGBoost. Разделение на train, validation и test позволило отделить подбор гиперпараметров от итоговой оценки качества и проверить модели на переобучение."
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>1. Постановка задачи и описание датасета</h1>')
    parts.append(paragraph_html(
        "Решается задача регрессии: по характеристикам клиента требуется предсказать непрерывную величину стоимости медицинской страховки (charges). Исходный датасет включает 1338 наблюдений и 7 столбцов, из которых шесть являются входными признаками, а один - целевой переменной. "
        + citation(9)
    ))
    parts.append(paragraph(
        "К количественным признакам относятся возраст клиента (age), индекс массы тела (bmi) и количество детей (children). Пол клиента (sex), факт курения (smoker) и регион проживания (region) представлены категориальными признаками. Фрагмент исходных данных приведен на рисунке 1."
    ))
    parts.append(fig(
        "raw_dataset_preview.png",
        "В исходном наборе одновременно присутствуют числовые и строковые значения. До передачи данных моделям категориальные столбцы необходимо преобразовать в числовую форму, сохранив содержательный смысл категорий.",
        "dataset-figure",
    ))
    parts.append(paragraph(
        f"Проверка целостности показала отсутствие пропущенных значений, поэтому заполнение данных не потребовалось. Вместе с тем выражение (df.duplicated().sum()) обнаружило {duplicate_count} полностью повторяющуюся строку. Она была удалена, и в дальнейшей работе использовались 1337 уникальных наблюдений."
    ))
    parts.append(code_fig(
        "code_data_quality",
        "missing_by_column = df.isna().sum()\n"
        "duplicate_count = df.duplicated().sum()\n"
        "df = df.drop_duplicates().reset_index(drop=True)"
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>2. Разведочный анализ и предобработка данных</h1>')
    parts.append('<h2>2.1. Выбор метрик качества</h2>')
    parts.append(paragraph_html(
        "До построения моделей были определены показатели, по которым будет оцениваться результат. Использовалось несколько метрик, поскольку одна величина не отражает все стороны ошибки. Коэффициент детерминации (R2) показывает долю вариации целевой переменной, объясненную моделью. Средняя абсолютная ошибка (MAE) характеризует типичное отклонение прогноза в долларах и сравнительно слабо зависит от единичных крупных промахов. "
        + citation(1, 2)
    ))
    parts.append(paragraph(
        "Среднеквадратическая ошибка (MSE) сильнее штрафует крупные ошибки, что важно для дорогих страховых случаев. Корень из среднеквадратической ошибки (RMSE) возвращает этот показатель в исходные единицы измерения. Средняя абсолютная процентная ошибка (MAPE) дополняет анализ относительной величиной ошибки, однако для небольших значений стоимости (charges) ее следует трактовать осторожно: деление на малое фактическое значение может непропорционально увеличить процентную ошибку."
    ))
    parts.append(
        '<div class="formula">R2 = 1 - Σ(y<sub>i</sub> - ŷ<sub>i</sub>)² / Σ(y<sub>i</sub> - ȳ)²</div>'
        '<div class="formula">MAE = (1 / n) Σ |y<sub>i</sub> - ŷ<sub>i</sub>|</div>'
        '<div class="formula">MSE = (1 / n) Σ (y<sub>i</sub> - ŷ<sub>i</sub>)²,&nbsp;&nbsp; RMSE = √MSE</div>'
        '<div class="formula">MAPE = (100% / n) Σ |(y<sub>i</sub> - ŷ<sub>i</sub>) / y<sub>i</sub>|</div>'
    )
    parts.append(paragraph(
        "Все метрики рассчитывались после обратного преобразования прогнозов в исходную шкалу стоимости. Их сопоставление на train, validation и test использовалось одновременно для ранжирования моделей и оценки устойчивости результатов."
    ))

    parts.append('<h2>2.2. Анализ распределения целевой переменной</h2>')
    parts.append(paragraph_html(
        "После удаления повторяющейся строки были рассчитаны описательные статистики стоимости страховки (charges), представленные в таблице 1. Затем форма распределения и высокие значения были исследованы графически средствами Matplotlib и Seaborn. "
        + citation(6, 7)
    ))
    stats = raw["charges"].agg(["mean", "var", "std", "min", "max"])
    target_median = float(raw["charges"].median())
    coefficient_of_variation = float(stats["std"] / stats["mean"] * 100)
    parts.append(table_html(
        ["Статистика", "Значение"],
        [
            ["Среднее", money(stats["mean"])],
            ["Медиана", money(target_median)],
            ["Мода (оценка KDE)", money(target_mode)],
            ["Дисперсия", money(stats["var"])],
            ["Стандартное отклонение", money(stats["std"])],
            ["Коэффициент вариации", f"{coefficient_of_variation:.1f}%"],
            ["Минимум", money(stats["min"])],
            ["Максимум", money(stats["max"])],
        ],
        "Таблица 1 - Описательные статистики целевой переменной (charges)",
    ))
    parts.append(paragraph_html(
        f"Стандартное отклонение ({money(stats['std'])} доллара) сопоставимо со средним значением ({money(stats['mean'])} доллара), а коэффициент вариации равен {coefficient_of_variation:.1f}%. Это характеризует высокий относительный разброс, но само по себе не определяет направление асимметрии. О правосторонней асимметрии свидетельствуют среднее, превышающее медиану ({money(target_median)} доллара), положительный коэффициент асимметрии и длинный хвост дорогих случаев до {money(stats['max'])} доллара. Поскольку все точные значения (charges) уникальны, мода непрерывной переменной оценивалась как максимум ядерной оценки плотности KDE; получено значение около {money(target_mode)} доллара. Такая оценка показывает наиболее плотную область распределения и зависит от выбранной ширины сглаживания. "
        + citation(10)
    ))
    parts.append(paragraph(
        f"Распределение представлено на рисунке {figure_numbers['charges_distribution_report.png']}: основная часть наблюдений сосредоточена в нижнем диапазоне, тогда как небольшая группа дорогих случаев формирует длинный правый хвост, что отражено в таблице 1 и выводах в п. 1."
    ))
    parts.append(fig(
        "charges_distribution_report.png",
        f"На гистограмме среднее ({money(stats['mean'])} доллара) расположено правее медианы ({money(target_median)} доллара). Коэффициент асимметрии равен {target_skew:.2f}: положительное значение соответствует длинному правому хвосту, поэтому исходное распределение имеет выраженную правостороннюю асимметрию.",
    ))
    parts.append(fig(
        "charges_boxplot_horizontal.png",
        "Диаграмма размаха выделяет группу высоких значений за верхней границей. Однако положение точки за границей диаграммы само по себе не означает ошибку: необходимо проверить, объясняются ли такие значения характеристиками клиентов.",
    ))
    parts.append(fig(
        "charges_boxplot_by_smoker.png",
        f"Стоимость выше 35 000 долларов имеют {outlier_count} объектов, или {outlier_share:.1f}% выборки. Среди них {outlier_smoker_share:.1f}% являются курильщиками. Поэтому большинство высоких значений отражает реальную закономерность для группы (smoker = yes), а не ошибки измерения; удалять их не следует.",
    ))

    parts.append('<h3>Логарифмирование целевой переменной</h3>')
    parts.append(paragraph_html(
        "Чтобы уменьшить влияние длинного правого хвоста, модель обучалась на логарифме стоимости. К исходной величине прибавлялась единица, что делает преобразование корректным и для нулевых значений. После прогноза выполнялось обратное преобразование, поэтому метрики рассчитывались в долларах. "
        + citation(5)
    ))
    parts.append(
        '<div class="formula">y<sub>log</sub> = ln(y + 1)</div>'
        '<div class="formula">ŷ = exp(ŷ<sub>log</sub>) - 1</div>'
    )
    parts.append(code_fig(
        "code_log_target",
        "df['charges_original'] = df['charges']\n"
        "df['charges'] = np.log1p(df['charges'])\n\n"
        "# обратное преобразование после прогноза\n"
        "y_pred = np.expm1(model.predict(X_test))"
    ))
    parts.append(paragraph(
        f"После логарифмирования коэффициент асимметрии снизился с {target_skew:.2f} до {log_target_skew:.2f}. Распределение стало значительно ближе к симметричному, а редкие дорогие полисы перестали чрезмерно доминировать в функции потерь."
    ))

    parts.append('<h2>2.3. Анализ линейной зависимости признаков</h2>')
    correlation_figure_number = figure_numbers["correlation_matrix_original_vs_log_target.png"]
    parts.append(labeled_figure(
        "correlation_matrix_original_target_large.png",
        f"{correlation_figure_number}, а",
        "Корреляционная матрица с исходной целевой переменной (charges)",
        "Для исходной стоимости наиболее заметна связь с возрастом клиента (age), однако ее коэффициент составляет только около 0,30. Умеренное значение коэффициента Пирсона может говорить о том, что связь нелинейна или зависит от взаимодействия возраста с курением и другими признаками, поскольку корреляция отражает только линейную составляющую зависимости. Между самими количественными признаками сильных линейных зависимостей не обнаружено.",
    ))
    parts.append(labeled_figure(
        "correlation_matrix_log_target_large.png",
        f"{correlation_figure_number}, б",
        "Корреляционная матрица с log(charges + 1)",
        "После логарифмирования связь возраста (age) с целевой переменной возрастает примерно до 0,53. Линейная связь с индексом массы тела (bmi) и количеством детей (children) остается слабой. Это не исключает их влияния, а может говорить о нелинейном характере связи, который далее проверяется моделями на основе деревьев.",
    ))

    parts.append('<h2>2.4. Кодирование и разделение данных</h2>')
    parts.append(paragraph_html(
        "Категориальные признаки (sex), (smoker) и (region) были преобразованы с помощью OneHotEncoder. Кодировщик обучался только на тренировочной части, что исключает перенос информации из validation и test в этап подготовки признаков. "
        + citation(2, 4)
    ))
    parts.append(paragraph(
        f"Стандартная стратификация для непрерывной целевой переменной невозможна. Поэтому упорядоченные значения (charges) были разбиты на пять интервалов (bins) с примерно одинаковым количеством объектов. Номер интервала объединялся с категорией (smoker), благодаря чему в train, validation и test сохранялись доля курильщиков и представительство разных диапазонов стоимости. Фрагмент кода разделения приведен на рисунке {figure_numbers['stratified_split_code.png']}."
    ))
    parts.append(fig("stratified_split_code.png", "", "code-figure"))
    parts.append(table_html(
        ["Выборка", "Количество объектов", "Доля"],
        [
            ["Train", str(len(train)), "60%"],
            ["Validation", str(len(val)), "20%"],
            ["Test", str(len(test)), "20%"],
        ],
        "Таблица 2 - Размеры итоговых выборок",
        "compact-table",
    ))
    parts.append(fig(
        "processed_dataset_preview.png",
        "После OneHotEncoding сформировано 8 входных признаков. Обозначения регионов: (r1) - northwest, (r2) - southeast, (r3) - southwest; регион northeast является базовой категорией и задается нулями во всех трех столбцах. Поле (charges) содержит логарифмированный таргет, а исходная стоимость хранится в (charges_original). Таким образом, после предобработки получен набор из 1337 уникальных наблюдений без пропусков; комбинированная стратификация обеспечила близкую структуру train, validation и test, а содержательно объяснимые дорогие случаи были сохранены.",
        "dataset-figure",
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>3. Базовая модель и проблема переобучения</h1>')
    parts.append('<h2>3.1. Обучение дерева решений</h2>')
    parts.append(paragraph_html(
        "Дерево решений формирует прогноз с помощью последовательных разбиений пространства признаков. В каждом узле алгоритм выбирает локально лучшее условие по критерию ошибки. Такая модель хорошо интерпретируется, но без ограничений может создавать листья для отдельных наблюдений и запоминать тренировочную выборку. "
        + citation(1, 2)
    ))
    parts.append(paragraph(
        f"Неограниченное дерево достигло глубины {unrestricted_tree.get_depth()}. На train оно получило R2 = {unrestricted_train['R2']:.4f} и RMSE = {money(unrestricted_train['RMSE'])}, тогда как на test R2 снизился до {unrestricted_test['R2']:.4f}, а RMSE вырос до {money(unrestricted_test['RMSE'])}. Разрыв R2 составил {unrestricted_train['R2'] - unrestricted_test['R2']:.4f}, что указывает на выраженное переобучение."
    ))
    parts.append('<div class="formula">Ошибка = Bias² + Variance + Noise</div>')
    parts.append(paragraph(
        "Неограниченное дерево имеет низкое смещение, но высокий разброс. Ограничение глубины и размеров листьев немного увеличивает смещение, зато снижает чувствительность к конкретному составу train."
    ))
    parts.append(fig(
        "decision_tree_structure_first_3_levels.png",
        "В каждом узле показано условие разбиения, а движение по ветвям последовательно сужает группу объектов. Для читаемости приведены только первые три уровня полного дерева.",
        "tree-figure",
    ))

    parts.append('<h2>3.2. Оптимизация гиперпараметров</h2>')
    parts.append(paragraph(
        "Для настройки применялись GridSearchCV и пятиблочная кросс-валидация KFold с перемешиванием. Двухэтапный поиск означает, что сначала широкая сетка с крупным шагом определяет перспективную область параметров, а затем уточненная сетка проверяет соседние значения с меньшим шагом. В данном случае после широкого поиска глубина и размеры узлов проверялись с шагом 1 рядом с найденным оптимумом. Validation в поиске гиперпараметров не участвовала и использовалась для независимой оценки модели."
    ))
    parts.append(code_fig(
        "code_tree_grid",
        "coarse_grid = {\n"
        "    'max_depth': [2, 4, 6, 8, 10, 12, 16, None],\n"
        "    'min_samples_split': [2, 10, 20, 40],\n"
        "    'min_samples_leaf': [1, 4, 8, 16],\n"
        "    'criterion': ['squared_error', 'absolute_error'],\n"
        "    'ccp_alpha': [0.0, 0.0001]\n"
        "}\n"
        "cv = KFold(n_splits=5, shuffle=True, random_state=42)\n"
        "grid = GridSearchCV(tree, coarse_grid, cv=cv, scoring=r2_original_scorer)"
    ))
    parts.append(paragraph(
        f"На рисунке {figure_numbers['decision_tree_best_params.png']} представлены значения гиперпараметров оптимизированного дерева."
    ))
    parts.append(fig(
        "decision_tree_best_params.png",
        f"После двухэтапного поиска выбрана глубина {tree_depth}. Параметры минимального размера узлов и стоимость отсечения (ccp_alpha) ограничивают дробление дерева и уменьшают его разброс.",
        "wide-figure",
    ))
    parts.append(fig(
        "decision_tree_overfitting_metrics.png",
        f"После оптимизации дерева R2 составил {tree_train['R2']:.4f} на train, {tree_val['R2']:.4f} на validation и {tree_test['R2']:.4f} на test. Ограничения почти не ухудшили test, но резко сократили разрыв с train, то есть уменьшили разброс модели.",
        "wide-figure",
    ))
    parts.append(fig(
        "decision_tree_residuals_test.png",
        f"Большинство остатков сосредоточено около нуля, поэтому модель улавливает основную зависимость. Для полисов с фактической стоимостью не выше 20 000 долларов стандартное отклонение остатков равно {money(tree_residual_zones['inexpensive_std'])}, а MAE - {money(tree_residual_zones['inexpensive_mae'])}. Для более дорогих полисов эти значения возрастают до {money(tree_residual_zones['expensive_std'])} и {money(tree_residual_zones['expensive_mae'])} соответственно. Следовательно, разброс ошибок действительно увеличивается в дорогом сегменте. Полосы точек возникают из-за одинакового прогноза для объектов одного листа.",
    ))
    parts.append(paragraph(
        "Двухэтапная настройка сохранила интерпретируемость дерева и заметно уменьшила переобучение. Однако чувствительность к дорогим случаям показала, что дальнейшее улучшение целесообразно искать в ансамблях, снижающих разброс модели."
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>4. Ансамблевые модели: Bagging и Random Forest</h1>')
    parts.append(paragraph_html(
        "В Bagging каждое дерево обучается на собственной bootstrap-подвыборке, после чего прогнозы усредняются. Bootstrap-подвыборка формируется случайным извлечением объектов исходной train-выборки с возвращением: один объект может встретиться несколько раз, а часть наблюдений не попадает в конкретную подвыборку. Поэтому деревья получают различающиеся обучающие данные, а усреднение их прогнозов снижает разброс ансамбля. Random Forest дополнительно ограничивает набор признаков, доступных при поиске разбиения. Это делает отдельные деревья менее похожими друг на друга и повышает устойчивость совместного прогноза. Для корректного сравнения использовались те же признаки, логарифмированный таргет и разбиение данных. "
        + citation(1, 2)
    ))

    parts.append('<h2>4.1. Оптимизация Bagging</h2>')
    parts.append(paragraph(
        "С помощью двухэтапной процедуры GridSearchCV были подобраны число деревьев (n_estimators), доля объектов (max_samples), доля признаков (max_features), глубина и минимальный размер узлов базового дерева. После широкого поиска на грубой сетке уточненная сетка строилась около лучших значений."
    ))
    parts.append(code_fig(
        "code_bagging_grid",
        "bagging_grid = {\n"
        "    'n_estimators': [50, 100, 200],\n"
        "    'max_samples': [0.6, 0.8, 1.0],\n"
        "    'max_features': [0.6, 0.8, 1.0],\n"
        "    'estimator__max_depth': [4, 6, 8, 10, None],\n"
        "    'estimator__min_samples_split': [2, 10, 20],\n"
        "    'estimator__min_samples_leaf': [1, 4, 8]\n"
        "}"
    ))
    parts.append(fig(
        "bagging_best_params.png",
        f"Итоговая конфигурация дерева определяет разнообразие базовых деревьев и объем данных для каждого из них. На тестовой выборке модель показала R2 = {bagging_test['R2']:.4f} и RMSE = {money(bagging_test['RMSE'])}, улучшив оба показателя относительно оптимизированного дерева.",
        "wide-figure",
    ))

    parts.append('<h2>4.2. Оптимизация Random Forest</h2>')
    parts.append(paragraph(
        "Random Forest оптимизировался по той же двухэтапной схеме. Помимо числа и глубины деревьев, параметр (max_features) задавал случайную долю признаков, доступных при разбиении, а (bootstrap) включал выбор объектов с возвращением."
    ))
    parts.append(code_fig(
        "code_random_forest_grid",
        "rf_grid = {\n"
        "    'n_estimators': [100, 200],\n"
        "    'max_depth': [4, 6, 8, 10, 12, None],\n"
        "    'min_samples_split': [2, 10, 20],\n"
        "    'min_samples_leaf': [1, 4, 8],\n"
        "    'max_features': ['sqrt', 0.7, 1.0],\n"
        "    'bootstrap': [True],\n"
        "    'ccp_alpha': [0.0, 0.0001]\n"
        "}"
    ))
    parts.append(fig(
        "random_forest_best_params.png",
        f"Параметр (max_features) управляет декорреляцией деревьев: слишком малое значение повышает разнообразие, но может ухудшить отдельные разбиения. На тестовой выборке Random Forest показал R2 = {forest_test['R2']:.4f} и RMSE = {money(forest_test['RMSE'])}.",
        "wide-figure",
    ))

    parts.append('<h2>4.3. Сравнение OOB-score и KFold</h2>')
    parts.append(paragraph_html(
        "OOB-score (out-of-bag score) - это внутренняя оценка качества bootstrap-ансамбля. Каждое дерево обучается на случайной выборке объектов с возвращением, поэтому часть наблюдений в его обучающую подвыборку не попадает. Для каждого объекта прогноз усредняется только по тем деревьям, которые этот объект не использовали при обучении; полученная оценка по смыслу напоминает проверку на отложенных данных и не требует выделять еще одну выборку. Чем ближе OOB R2 к результатам KFold и test, тем устойчивее выглядит оценка качества модели. Для одиночного дерева OOB-score не определен, поскольку оно не является bootstrap-композицией. "
        + citation(2)
    ))
    parts.append(code_fig(
        "code_oob_models",
        "bagging_oob = BaggingRegressor(\n"
        "    estimator=best_tree, n_estimators=best_n,\n"
        "    bootstrap=True, oob_score=True, random_state=42\n"
        ")\n\n"
        "rf_oob = RandomForestRegressor(\n"
        "    n_estimators=best_n, bootstrap=True,\n"
        "    oob_score=True, random_state=42\n"
        ")"
    ))
    parts.append(fig(
        "oob_kfold_test_comparison.png",
        "Для Bagging OOB R2 равен 0,8628 против 0,8478 по KFold, для Random Forest - 0,8637 против 0,8466. Тестовые значения 0,8475 и 0,8485 близки к KFold, поэтому общий вывод о качестве не зависит от одной процедуры валидации.",
        "wide-figure",
    ))
    parts.append(fig(
        "random_forest_feature_importance.png",
        "Наибольший вклад в Random Forest имеют факт курения (smoker_yes) и возраст (age). Далее следуют индекс массы тела (bmi) и количество детей (children), причем оба признака дают заметный, но существенно меньший вклад. Пол и регион влияют слабо. Результат согласуется с предварительным анализом данных.",
    ))
    parts.append(paragraph(
        "Модели Bagging и Random Forest улучшили R2 и RMSE относительно оптимизированного дерева, сохранив близкие результаты на train, validation и test. Небольшая разница между ансамблями может объясняться сильным влиянием курения: даже при случайном выборе признаков модели часто приходят к похожим ключевым разбиениям."
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>5. Градиентный бустинг: XGBoost</h1>')
    parts.append(paragraph_html(
        "В отличие от Bagging и Random Forest, градиентный бустинг строит деревья последовательно: каждое новое дерево направлено на исправление ошибок уже сформированного ансамбля. В работе использовалась модель XGBRegressor из библиотеки XGBoost. "
        + citation(1, 3)
    ))
    parts.append(paragraph(
        "На каждом шаге вычисляются псевдоостатки, задающие направление уменьшения функции потерь. Для квадратичной ошибки они совпадают с обычными остатками. Новое дерево приближает эти значения, а величина его вклада регулируется скоростью обучения (learning_rate)."
    ))
    parts.append(
        '<div class="formula">r<sub>i</sub><sup>(m)</sup> = -∂L(y<sub>i</sub>, F(x<sub>i</sub>)) / ∂F(x<sub>i</sub>)</div>'
        '<div class="formula">F<sub>m</sub>(x) = F<sub>m-1</sub>(x) + ηh<sub>m</sub>(x)</div>'
    )
    parts.append('<h2>5.1. Двухэтапная оптимизация гиперпараметров</h2>')
    parts.append(paragraph(
        f"Сначала RandomizedSearchCV проверил 60 комбинаций на пяти блоках KFold, затем еще 50 комбинаций были исследованы около лучшего решения. Средний CV R2 вырос с {xgb_results.get('coarse_best_score', float('nan')):.4f} до {xgb_results.get('refined_best_score', float('nan')):.4f}; наилучшее качество на validation достигнуто примерно после {xgb_results.get('best_stage', 'NA')} деревьев."
    ))
    parts.append(code_fig(
        "code_xgboost_grid",
        "xgb_grid = {\n"
        "    'n_estimators': [150, 250, 400, 600, 800],\n"
        "    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08],\n"
        "    'max_depth': [2, 3, 4, 5],\n"
        "    'min_child_weight': [1, 3, 5, 8],\n"
        "    'subsample': [0.7, 0.8, 0.9, 1.0],\n"
        "    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],\n"
        "    'reg_lambda': [0.5, 1, 3, 5, 10],\n"
        "    'reg_alpha': [0, 0.01, 0.1, 0.5]\n"
        "}"
    ))
    parts.append(fig(
        "xgboost_best_params.png",
        "Небольшая скорость обучения (learning_rate = 0,01), глубина 3 и регуляризация обеспечивают постепенное улучшение прогноза без чрезмерно сложных отдельных деревьев.",
        "wide-figure",
    ))
    parts.append(fig(
        "xgboost_learning_curve.png",
        f"На увеличенном участке показаны контрольные значения Validation R2: 400 деревьев - {learning_curve_scores[400]:.4f}, 500 - {learning_curve_scores[500]:.4f}, 600 - {learning_curve_scores[600]:.4f}, {selected_learning_stage} - {learning_curve_scores[selected_learning_stage]:.4f}, 700 - {learning_curve_scores[700]:.4f}. Качество растет до {selected_learning_stage} деревьев, где достигается максимум, а затем немного снижается; поэтому выбрано именно {selected_learning_stage} деревьев.",
    ))
    parts.append(fig(
        "xgboost_residuals_test.png",
        f"Облако остатков в основном центрировано около нуля, поэтому общего систематического смещения не наблюдается. Для полисов с фактической стоимостью не выше 20 000 долларов стандартное отклонение остатков равно {money(xgb_residual_zones['inexpensive_std'])}, а MAE - {money(xgb_residual_zones['inexpensive_mae'])}. В более дорогом сегменте показатели возрастают до {money(xgb_residual_zones['expensive_std'])} и {money(xgb_residual_zones['expensive_mae'])}. Следовательно, XGBoost также ошибается менее стабильно на дорогих полисах, хотя их разброс ниже, чем у оптимизированного дерева.",
    ))
    parts.append(fig(
        "xgboost_feature_importance.png",
        "Признак курения (smoker_yes) формирует около 70% суммарной важности. Далее следуют возраст (age), количество детей (children) и индекс массы тела (bmi). Это подтверждает гипотезу EDA о сильном влиянии курения.",
    ))
    parts.append(paragraph(
        f"На test XGBoost достиг R2 = {xgb_test['R2']:.4f} и RMSE = {money(xgb_test['RMSE'])}, показав лучший результат по этим двум метрикам. Близкие значения R2 на train ({xgb_train['R2']:.4f}), validation ({xgb_val['R2']:.4f}) и test не указывают на выраженное переобучение."
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>6. Сравнительный анализ моделей</h1>')
    parts.append(paragraph(
        "Итоговое сравнение выполнялось на неизменных выборках. Test не участвовала ни в обучении, ни в подборе гиперпараметров и поэтому служила основной оценкой обобщающей способности. Помимо метрик анализировались остатки, фактические и предсказанные значения, важность признаков, SHAP и скорость инференса."
    ))
    metric_rows: list[list[str]] = []
    model_order = ["Оптимизированное дерево", "Bagging", "Random Forest", "XGBoost"]
    split_order = ["Train", "Validation", "Test"]
    for model_name in model_order:
        for split_name in split_order:
            row = metric_by_model_split.loc[(model_name, split_name)]
            metric_rows.append([
                model_name,
                split_name,
                f"{row['R2']:.4f}",
                money(row["MAE"]),
                f"{row['MSE']:.2e}",
                money(row["RMSE"]),
                f"{row['MAPE']:.2f}%",
            ])
    parts.append(table_html(
        ["Модель", "Выборка", "R2", "MAE", "MSE", "RMSE", "MAPE"],
        metric_rows,
        "Таблица 3 - Метрики моделей на train, validation и test",
        "metrics-table",
    ))
    parts.append(paragraph(
        "Таблица 3 позволяет одновременно сравнить итоговое качество и разрыв между выборками. Все значения рассчитаны в исходной шкале после преобразования exp(prediction) - 1."
    ))
    parts.append(fig(
        "comparison_r2_train_test.png",
        "У всех оптимизированных моделей разрыв между train и test остается умеренным. Несмотря на то что метрики соизмеримы, наименьшая разница между метрикой R2 на обучающей и тестовой выборках наблюдается у модели XGBoost, тогда как у Random Forest различие между train и test немного больше, чем у остальных ансамблей.",
    ))
    parts.append(fig(
        "comparison_errors_models.png",
        "RMSE находится в узком диапазоне примерно от 4 300 до 4 565 долларов. Все модели используют одни признаки и сталкиваются с одинаковыми редкими дорогими случаями, которые сильнее всего влияют на квадрат ошибки. У оптимизированного дерева MAE ниже, но RMSE выше, то есть большинство прогнозов близки к факту, однако отдельные крупные промахи сильнее ухудшают результат.",
    ))
    parts.append(fig(
        "comparison_actual_vs_predicted_all_models.png",
        "Чем ближе точки к диагонали, тем точнее прогноз. XGBoost формирует наиболее компактное облако, но в верхнем диапазоне стоимости все модели чаще отклоняются от идеальной линии.",
        "wide-figure",
    ))
    parts.append(fig(
        "comparison_residuals_all_models.png",
        "У ансамблей облака остатков более сглажены, чем у одиночного дерева, а постоянного смещения относительно нуля не видно. Расширение облака справа подтверждает, что дорогие полисы остаются общей сложной областью.",
        "wide-figure",
    ))
    parts.append(fig(
        "comparison_feature_importance_all_models.png",
        "Во всех ансамблях главным фактором остается курение (smoker_yes), далее обычно следуют возраст (age) и индекс массы тела (bmi). Пол и регион практически не влияют на целевую переменную, что согласуется с предварительным анализом.",
        "wide-figure",
    ))
    parts.append(paragraph_html(
        "SHAP (SHapley Additive exPlanations) - метод интерпретации, который раскладывает отдельный прогноз модели на базовое значение и вклады признаков. Положительное SHAP-значение повышает прогноз относительно базового уровня, отрицательное - понижает, а абсолютная величина показывает силу влияния признака для конкретного объекта. Обобщение таких значений по всей выборке позволяет оценить глобальную важность и направление воздействия признаков. "
        + citation(8)
    ))
    parts.append(fig(
        "comparison_shap_summary_random_forest.png",
        "SHAP показывает не только силу, но и направление влияния. Курение и больший возраст чаще сдвигают прогноз log(charges + 1) вверх, что дополняет встроенную важность признаков содержательной интерпретацией.",
        "wide-figure",
    ))
    parts.append(fig(
        "comparison_inference_speed.png",
        "Одиночное дерево прогнозирует быстрее всего, XGBoost занимает промежуточное положение, а Bagging и Random Forest требуют больше времени из-за вычисления прогнозов множества деревьев. Для данного небольшого датасета все измеренные значения остаются практически приемлемыми.",
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>Заключение</h1>')
    parts.append(paragraph(
        "В ходе работы был исследован набор из 1338 наблюдений для прогнозирования стоимости медицинской страховки (charges). Пропущенных значений не обнаружено, одна полностью повторяющаяся строка удалена. Категориальные признаки преобразованы методом OneHotEncoding, а данные разделены на train, validation и test с комбинированной стратификацией по интервалам стоимости и признаку курения (smoker)."
    ))
    parts.append(paragraph(
        f"EDA выявил правостороннюю асимметрию целевой переменной: ее коэффициент составил {target_skew:.2f}. Среди {outlier_count} полисов дороже 35 000 долларов доля курильщиков достигла {outlier_smoker_share:.1f}%, поэтому высокие значения были признаны содержательно объяснимыми и сохранены. Логарифмирование уменьшило асимметрию до {log_target_skew:.2f}."
    ))
    parts.append(paragraph(
        f"На test оптимизированное дерево получило R2 = {tree_test['R2']:.4f}, Bagging - {bagging_test['R2']:.4f}, Random Forest - {forest_test['R2']:.4f}, а XGBoost - {xgb_test['R2']:.4f}. Лучшей по R2 и RMSE стала модель XGBoost: RMSE составил {money(xgb_test['RMSE'])}, MAE - {money(xgb_test['MAE'])}. При этом оптимизированное дерево показало меньшие MAE ({money(tree_test['MAE'])}) и MAPE ({tree_test['MAPE']:.2f}%), поэтому выбор модели зависит от того, насколько критичны отдельные крупные ошибки."
    ))
    parts.append(paragraph(
        f"Для XGBoost значения R2 на train, validation и test равны {xgb_train['R2']:.4f}, {xgb_val['R2']:.4f} и {xgb_test['R2']:.4f}. Небольшой разрыв не указывает на выраженное переобучение, а test R2 означает, что модель объясняет около {xgb_test['R2'] * 100:.1f}% вариации стоимости на ранее не использованных данных. Основная часть оставшейся ошибки связана с наиболее дорогими полисами."
    ))
    parts.append(paragraph(
        "Интерпретация лучшей модели согласуется с EDA. Признак курения (smoker_yes) формирует около 70% встроенной важности XGBoost, подтверждая концентрацию дорогих полисов среди курильщиков. Возраст (age) также остается значимым фактором и соответствует наиболее сильной корреляции с log(charges + 1). Индекс массы тела (bmi) и количество детей (children) дают меньший, преимущественно нелинейный вклад, а пол (sex) и регион (region) почти не влияют на прогноз."
    ))
    parts.append(paragraph(
        "В последующих работах настройку XGBoost можно сделать более глубокой. Для этого целесообразно увеличить число проверяемых комбинаций и расширить диапазоны количества деревьев (n_estimators), скорости обучения (learning_rate), глубины (max_depth), минимального веса дочернего узла (min_child_weight), долей объектов и признаков (subsample, colsample_bytree), а также коэффициентов регуляризации (reg_alpha, reg_lambda). После широкого поиска перспективную область следует исследовать с меньшим шагом либо применить байесовскую оптимизацию. Раннее прекращение обучения (early stopping) позволит выбирать число деревьев по качеству на validation, а повторная или вложенная кросс-валидация снизит зависимость результата от одного разбиения данных."
    ))
    parts.append(paragraph(
        "Дополнительный резерв качества связан с данными и постановкой задачи. Полезно добавить сведения о хронических заболеваниях, типе страхового плана, уровне дохода и истории обращений, проверить взаимодействия возраста (age), курения (smoker) и индекса массы тела (bmi), а также отдельно настроить функцию потерь для дорогих полисов. Можно сравнить обучение на исходной и логарифмической шкале, робастные функции потерь и интервальные прогнозы. Каждое улучшение необходимо подтверждать на одной и той же отложенной тестовой выборке, а итоговую модель дополнительно проверять на внешних или более поздних данных."
    ))
    parts.append(paragraph(
        "Итоговой моделью по совокупности R2 и RMSE выбран XGBoost. Оптимизированное дерево остается наиболее простым и быстрым решением, Random Forest дает близкое качество и удобен для OOB- и SHAP-анализа, однако XGBoost лучше всего сокращает крупные ошибки и обеспечивает наиболее высокий test R2."
    ))
    parts.append("</section>")

    parts.append('<section class="chapter"><h1>Список использованных источников</h1><ol class="source-list">')
    sources = [
        (
            "Яндекс Образование. Учебник по машинному обучению: решающие деревья, ансамбли, градиентный бустинг, метрики и кросс-валидация.",
            "https://education.yandex.ru/handbook/ml",
        ),
        (
            "Scikit-learn documentation. DecisionTreeRegressor, BaggingRegressor, RandomForestRegressor, OneHotEncoder, GridSearchCV и RandomizedSearchCV.",
            "https://scikit-learn.org/stable/",
        ),
        (
            "XGBoost documentation. XGBRegressor and gradient boosted decision trees.",
            "https://xgboost.readthedocs.io/en/stable/",
        ),
        (
            "Pandas documentation. Работа с табличными данными и расчет описательных статистик.",
            "https://pandas.pydata.org/docs/",
        ),
        (
            "NumPy documentation. Численные преобразования, включая log1p и expm1.",
            "https://numpy.org/doc/stable/",
        ),
        (
            "Matplotlib documentation. Построение и оформление графиков.",
            "https://matplotlib.org/stable/",
        ),
        (
            "Seaborn documentation. Статистическая визуализация данных.",
            "https://seaborn.pydata.org/",
        ),
        (
            "SHAP documentation. Интерпретация моделей машинного обучения на основе SHAP-значений.",
            "https://shap.readthedocs.io/en/latest/",
        ),
        (
            "Medical Cost Personal Datasets. Набор данных о стоимости медицинской страховки.",
            "https://www.kaggle.com/datasets/mirichoi0218/insurance",
        ),
        (
            "SciPy documentation. Ядерная оценка плотности распределения gaussian_kde.",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html",
        ),
    ]
    for number, (description, url) in enumerate(sources, start=1):
        parts.append(
            f'<li id="source-{number}">{e(description)} URL: '
            f'<a href="{e(url)}">{e(url)}</a>.</li>'
        )
    parts.append("</ol></section>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-map-from", type=Path)
    args = parser.parse_args()

    page_map = detect_page_map(args.page_map_from)
    report_html = build_report(page_map)
    HTML_PATH.write_text(report_html, encoding="utf-8")
    print(f"HTML saved: {HTML_PATH}")
    if page_map:
        print(f"TOC page map: {page_map}")


if __name__ == "__main__":
    main()
