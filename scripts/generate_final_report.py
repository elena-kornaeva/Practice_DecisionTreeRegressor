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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = REPORTS_DIR / "assets"
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
    ASSETS_DIR.mkdir(exist_ok=True)


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


def calculate_split_metrics(
    models: dict[str, object], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(val)
    x_test, y_test = split_xy(test)

    rows = []
    for model_name, model in models.items():
        for split_name, x, y in [
            ("Train", x_train, y_train),
            ("Validation", x_val, y_val),
            ("Test", x_test, y_test),
        ]:
            y_pred = predict_original_scale(model, x)
            mse = mean_squared_error(y, y_pred)
            rows.append(
                {
                    "Модель": model_name,
                    "Выборка": split_name,
                    "R2": r2_score(y, y_pred),
                    "MAE": mean_absolute_error(y, y_pred),
                    "MSE": mse,
                    "RMSE": math.sqrt(mse),
                    "MAPE": mean_absolute_percentage_error(y, y_pred) * 100,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        REPORTS_DIR / "train_validation_test_metrics_table.csv",
        index=False,
        encoding="utf-8",
    )
    metrics_df[metrics_df["Выборка"].isin(["Train", "Test"])].to_csv(
        REPORTS_DIR / "train_test_metrics_table.csv",
        index=False,
        encoding="utf-8",
    )
    return metrics_df


def calculate_unrestricted_tree_metrics(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[DecisionTreeRegressor, pd.DataFrame]:
    x_train = train.drop(columns=["charges", "charges_original"])
    model = DecisionTreeRegressor(random_state=42)
    model.fit(x_train, train["charges"])

    rows = []
    for split_name, data in [("Train", train), ("Validation", val), ("Test", test)]:
        x, y = split_xy(data)
        y_pred = predict_original_scale(model, x)
        mse = mean_squared_error(y, y_pred)
        rows.append(
            {
                "Модель": "Дерево без ограничений",
                "Выборка": split_name,
                "R2": r2_score(y, y_pred),
                "MAE": mean_absolute_error(y, y_pred),
                "MSE": mse,
                "RMSE": math.sqrt(mse),
                "MAPE": mean_absolute_percentage_error(y, y_pred) * 100,
            }
        )
    return model, pd.DataFrame(rows)


def save_table_image(
    df: pd.DataFrame,
    path: Path,
    title: str,
    col_widths: list[float] | None = None,
    fontsize: int = 9,
    figsize: tuple[float, float] | None = None,
    table_scale_y: float = 1.35,
) -> None:
    default_size = (8.2, max(2.0, 0.45 * len(df) + 1.2))
    fig, ax = plt.subplots(figsize=figsize or default_size)
    ax.axis("off")
    if title:
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
    table.scale(1, table_scale_y)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#4d4d4d")
        if row == 0:
            cell.set_facecolor("#dbead2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f7fbf4" if row % 2 else "#ffffff")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_new_figures(
    raw: pd.DataFrame,
    train: pd.DataFrame,
    metrics_df: pd.DataFrame,
    unrestricted_metrics: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    raw_preview = raw.head(7).copy()
    raw_preview["charges"] = raw_preview["charges"].round(2)
    paths["raw_preview"] = FIGURES_DIR / "raw_dataset_preview.png"
    save_table_image(
        raw_preview,
        paths["raw_preview"],
        "",
        col_widths=[0.08, 0.10, 0.10, 0.11, 0.11, 0.18, 0.16],
        fontsize=8,
        figsize=(8.2, 2.0),
        table_scale_y=1.15,
    )

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

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(raw["charges"], bins=30, color="#74a9cf", edgecolor="white", alpha=0.9)
    ax.axvline(raw["charges"].mean(), color="#d95f0e", linestyle="--", linewidth=1.7, label="Среднее")
    ax.axvline(raw["charges"].median(), color="#238b45", linestyle=":", linewidth=1.9, label="Медиана")
    ax.set_xlabel("Стоимость страховки (charges), $")
    ax.set_ylabel("Количество наблюдений")
    ax.set_title("Распределение целевой переменной (charges)")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    paths["charges_distribution"] = FIGURES_DIR / "charges_distribution_report.png"
    fig.savefig(paths["charges_distribution"], bbox_inches="tight")
    plt.close(fig)

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
    preview = preview.rename(
        columns={
            "region_northwest": "r1",
            "region_southeast": "r2",
            "region_southwest": "r3",
        }
    )
    preview = preview.round(3)
    paths["processed_preview"] = FIGURES_DIR / "processed_dataset_preview.png"
    save_table_image(
        preview,
        paths["processed_preview"],
        "",
        fontsize=7,
        figsize=(8.2, 1.75),
        table_scale_y=1.15,
    )

    table_df = metrics_df.copy()
    table_df["R2"] = table_df["R2"].map(metric)
    table_df["MAE"] = table_df["MAE"].map(money)
    table_df["MSE"] = table_df["MSE"].map(lambda value: f"{value:.2e}")
    table_df["RMSE"] = table_df["RMSE"].map(money)
    table_df["MAPE"] = table_df["MAPE"].map(lambda value: f"{value:.2f}%")
    paths["split_metrics"] = FIGURES_DIR / "train_validation_test_metrics_table.png"
    save_table_image(
        table_df,
        paths["split_metrics"],
        "Метрики моделей на train, validation и test",
        col_widths=[0.22, 0.12, 0.08, 0.12, 0.12, 0.12, 0.10],
        fontsize=7,
    )

    tree_comparison = pd.concat(
        [
            unrestricted_metrics,
            metrics_df[metrics_df["Модель"] == "Оптимизированное дерево"],
        ],
        ignore_index=True,
    )[["Модель", "Выборка", "R2", "MAE", "RMSE"]]
    tree_comparison["R2"] = tree_comparison["R2"].map(metric)
    tree_comparison["MAE"] = tree_comparison["MAE"].map(money)
    tree_comparison["RMSE"] = tree_comparison["RMSE"].map(money)
    paths["tree_overfitting"] = FIGURES_DIR / "decision_tree_overfitting_metrics.png"
    save_table_image(
        tree_comparison,
        paths["tree_overfitting"],
        "Переобучение дерева до и после оптимизации",
        col_widths=[0.28, 0.15, 0.10, 0.15, 0.15],
        fontsize=8,
    )

    oob_table = pd.DataFrame(
        [
            ["Оптимизированное дерево", "-", "0.8319", "0.8399"],
            ["Bagging", "0.8628", "0.8478", "0.8475"],
            ["Random Forest", "0.8637", "0.8466", "0.8485"],
        ],
        columns=["Модель", "OOB R2", "KFold R2", "Test R2"],
    )
    paths["oob_comparison"] = FIGURES_DIR / "oob_kfold_test_comparison.png"
    save_table_image(
        oob_table,
        paths["oob_comparison"],
        "Сравнение OOB, KFold и test",
        col_widths=[0.30, 0.15, 0.15, 0.15],
        fontsize=8,
    )

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
    train_bars = ax.bar(x - width / 2, r2_pivot["Train"], width, label="Train", color="#74a9cf")
    test_bars = ax.bar(x + width / 2, r2_pivot["Test"], width, label="Test", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(r2_pivot.index, rotation=12, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Сравнение моделей по R2 на train и test")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.75, 0.92)
    ax.bar_label(train_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(test_bars, fmt="%.3f", padding=2, fontsize=8)
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
    fig.text(0.5, 0.025, str(number), ha="center", va="bottom", fontsize=10)


def wrap_code_block(code: str, width: int = 68) -> str:
    wrapped_lines: list[str] = []
    for line in code.splitlines():
        if not line or len(line) <= width:
            wrapped_lines.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip())]
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=indent + "    ",
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
        )
    return "\n".join(wrapped_lines)


def page_text_fits(fig: plt.Figure, bottom: float = 0.055) -> bool:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()

    for artist in fig.texts:
        if not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer).transformed(inverse)
        is_page_number = artist.get_position()[1] < 0.04
        if bbox.x0 < 0.045 or bbox.x1 > 0.955 or bbox.y1 > 0.975:
            return False
        if not is_page_number and bbox.y0 < bottom:
            return False
    return True


def save_report_figure(pdf: PdfPages, fig: plt.Figure, page_no: int) -> None:
    pdf.savefig(fig)
    preview_dir_value = os.environ.get("REPORT_PREVIEW_DIR")
    if preview_dir_value:
        preview_dir = Path(preview_dir_value)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"page_{page_no:03d}.png", dpi=110)


def add_title_page(pdf: PdfPages, page_no: int) -> int:
    fig = plt.figure(figsize=A4)

    logo = mpimg.imread(ASSETS_DIR / "innopolis_university_logo.png")
    logo_ax = fig.add_axes([0.43, 0.855, 0.14, 0.099])
    logo_ax.imshow(logo)
    logo_ax.axis("off")

    fig.text(
        0.5,
        0.79,
        "ОТЧЕТ ПО ПРАКТИКЕ",
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(0.5, 0.675, "Тема работы", ha="center", va="top", fontsize=14)
    fig.text(
        0.5,
        0.62,
        "Построение и анализ ансамблевых моделей\n"
        "на основе решающих деревьев: регрессия",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        linespacing=1.35,
    )

    fig.text(
        0.56,
        0.40,
        "Выполнили:\n"
        "Камалов Булат\n"
        "Умаханов Мансур",
        ha="left",
        va="top",
        fontsize=13,
        linespacing=1.55,
    )
    fig.text(
        0.56,
        0.25,
        "Научный руководитель:\n"
        "Елена Корнаева Петровна",
        ha="left",
        va="top",
        fontsize=13,
        linespacing=1.55,
    )
    fig.text(0.5, 0.115, "Университет Иннополис", ha="center", va="bottom", fontsize=12)
    fig.text(0.5, 0.075, "2026", ha="center", va="bottom", fontsize=12)

    save_report_figure(pdf, fig, page_no)
    plt.close(fig)
    return page_no + 1


def add_text_page(
    pdf: PdfPages,
    page_no: int,
    title: str,
    paragraphs: list[str],
    code: str | None = None,
    formulas: list[str] | None = None,
) -> int:
    body_width = 68
    body_size = 12.0
    body_gap = 0.0315
    code_size = 9.4
    prepared_code = wrap_code_block(code, width=68) if code else None
    title_lines = textwrap.wrap(title, width=44)
    content_top = 0.95 - 0.068 - max(0, len(title_lines) - 1) * 0.040
    content_bottom = 0.075
    available_height = content_top - content_bottom

    blocks: list[tuple[str, object, float]] = []
    for paragraph in paragraphs:
        wrapped_count = 0
        for part in paragraph.split("\n"):
            wrapped_count += max(1, len(textwrap.wrap(part, width=body_width)))
        blocks.append(("paragraph", paragraph, wrapped_count * body_gap + 0.022))
    if formulas:
        blocks.append(("formulas", formulas, len(formulas) * 0.055 + 0.012))
    if prepared_code:
        code_lines = prepared_code.count("\n") + 1
        blocks.append(("code", prepared_code, code_lines * 0.0155 + 0.030))

    pages: list[list[tuple[str, object, float]]] = []
    current_page: list[tuple[str, object, float]] = []
    current_height = 0.0
    for block in blocks:
        block_height = block[2]
        if current_page and current_height + block_height > available_height:
            pages.append(current_page)
            current_page = []
            current_height = 0.0
        current_page.append(block)
        current_height += block_height
    if current_page or not pages:
        pages.append(current_page)

    for page_index, page_blocks in enumerate(pages):
        fig = plt.figure(figsize=A4)
        if page_index == 0:
            fig.text(
                0.5,
                0.95,
                "\n".join(title_lines),
                ha="center",
                va="top",
                fontsize=17,
                fontweight="bold",
                linespacing=1.15,
            )
            y = content_top
        else:
            # На страницах-продолжениях заголовок главы не повторяется.
            y = 0.95
        for block_type, block_content, _height in page_blocks:
            if block_type == "paragraph":
                y = draw_wrapped(
                    fig,
                    str(block_content),
                    0.10,
                    y,
                    width=body_width,
                    size=body_size,
                    line_gap=body_gap,
                )
                y -= 0.010
            elif block_type == "formulas":
                y -= 0.006
                for formula in block_content:
                    fig.text(0.5, y, formula, ha="center", va="top", fontsize=13)
                    y -= 0.055
            elif block_type == "code":
                y -= 0.008
                fig.text(
                    0.10,
                    y,
                    str(block_content),
                    fontsize=code_size,
                    fontfamily="DejaVu Sans Mono",
                    va="top",
                    linespacing=1.25,
                    bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f2f2f2", "edgecolor": "#bdbdbd"},
                )
        add_page_number(fig, page_no)
        if not page_text_fits(fig, bottom=0.06):
            print(f"Layout warning on text page {page_no}: {title}")
        save_report_figure(pdf, fig, page_no)
        plt.close(fig)
        page_no += 1
    return page_no


def add_image_page(
    pdf: PdfPages,
    page_no: int,
    section_title: str,
    image_path: Path,
    caption: str,
    body: str | None = None,
    show_section_title: bool = False,
    after_body: list[str] | None = None,
) -> int:
    caption_title, caption_explanation = split_caption(caption)
    fig = plt.figure(figsize=A4)
    y = 0.95
    if show_section_title:
        section_title_lines = textwrap.wrap(section_title, width=42)
        fig.text(
            0.5,
            y,
            "\n".join(section_title_lines),
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
            linespacing=1.15,
        )
        y -= 0.064 + max(0, len(section_title_lines) - 1) * 0.040
    if body:
        y = draw_wrapped(fig, body, 0.10, y, width=68, size=11.8, line_gap=0.031)
        y -= 0.015

    caption_size = 11.2
    explanation_size = 11.5
    caption_gap = 0.030
    title_lines = max(1, len(textwrap.wrap(caption_title, width=68)))
    explanation_lines = len(textwrap.wrap(caption_explanation, width=68)) if caption_explanation else 0
    caption_height = title_lines * caption_gap
    if explanation_lines:
        caption_height += 0.010 + explanation_lines * caption_gap
    after_body_height = 0.0
    if after_body:
        after_body_lines = sum(
            max(1, len(textwrap.wrap(part, width=68)))
            for paragraph in after_body
            for part in paragraph.split("\n")
        )
        after_body_height = 0.020 + after_body_lines * caption_gap + len(after_body) * 0.010
        caption_height += after_body_height

    img = mpimg.imread(image_path)
    img_h, img_w = img.shape[:2]
    max_w = 0.80
    max_h = min(0.50, max(0.20, y - caption_height - 0.125))
    aspect = img_w / img_h
    if aspect >= max_w / max_h:
        w = max_w
        h = w / aspect
    else:
        h = max_h
        w = h * aspect
    left = (1 - w) / 2
    bottom = y - h
    ax = fig.add_axes([left, bottom, w, h])
    ax.imshow(img)
    ax.axis("off")

    caption_y = bottom - 0.035
    explanation_y = draw_wrapped(
        fig,
        caption_title,
        0.10,
        caption_y,
        width=68,
        size=caption_size,
        line_gap=caption_gap,
    )
    if caption_explanation:
        explanation_y -= 0.008
        explanation_y = draw_wrapped(
            fig,
            caption_explanation,
            0.10,
            explanation_y,
            width=68,
            size=explanation_size,
            line_gap=caption_gap,
        )
    if after_body:
        explanation_y -= 0.015
        for paragraph in after_body:
            explanation_y = draw_wrapped(
                fig,
                paragraph,
                0.10,
                explanation_y,
                width=68,
                size=explanation_size,
                line_gap=caption_gap,
            )
            explanation_y -= 0.010
    add_page_number(fig, page_no)
    if not page_text_fits(fig, bottom=0.05):
        print(f"Layout warning on image page {page_no}: {section_title}")
    save_report_figure(pdf, fig, page_no)
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


def model_metrics_sentence(metrics_df: pd.DataFrame, model_name: str) -> str:
    selected = metrics_df[metrics_df["Модель"] == model_name].set_index("Выборка")
    parts = []
    for split_name, label in [("Train", "train"), ("Validation", "validation"), ("Test", "test")]:
        values = selected.loc[split_name]
        parts.append(
            f"{label}: R2 = {values['R2']:.4f}, MAE = {money(values['MAE'])}, "
            f"RMSE = {money(values['RMSE'])}, MAPE = {values['MAPE']:.2f}%"
        )
    return "; ".join(parts) + "."


def main() -> None:
    ensure_dirs()
    raw, train, val, test = load_data()
    models = load_models()
    metrics_df = calculate_split_metrics(models, train, val, test)
    unrestricted_tree, unrestricted_metrics = calculate_unrestricted_tree_metrics(train, val, test)
    figure_paths = make_new_figures(raw, train, metrics_df, unrestricted_metrics)

    outliers_35000 = raw[raw["charges"] > 35000]
    outlier_count = len(outliers_35000)
    outlier_share = outlier_count / len(raw) * 100
    outlier_smoker_share = (outliers_35000["smoker"] == "yes").mean() * 100
    duplicate_count = int(raw.duplicated().sum())
    target_skew = raw["charges"].skew()
    log_target_skew = np.log1p(raw["charges"]).skew()

    opt_tree = models["Оптимизированное дерево"]
    tree_depth = opt_tree.get_depth() if hasattr(opt_tree, "get_depth") else None
    tree_metrics = metrics_df[metrics_df["Модель"] == "Оптимизированное дерево"].set_index("Выборка")
    tree_train_r2 = tree_metrics.loc["Train", "R2"]
    tree_test_r2 = tree_metrics.loc["Test", "R2"]
    tree_train_rmse = tree_metrics.loc["Train", "RMSE"]
    tree_test_rmse = tree_metrics.loc["Test", "RMSE"]
    tree_r2_gap = tree_train_r2 - tree_test_r2

    unrestricted_by_split = unrestricted_metrics.set_index("Выборка")
    unrestricted_train_r2 = unrestricted_by_split.loc["Train", "R2"]
    unrestricted_test_r2 = unrestricted_by_split.loc["Test", "R2"]
    unrestricted_train_rmse = unrestricted_by_split.loc["Train", "RMSE"]
    unrestricted_test_rmse = unrestricted_by_split.loc["Test", "RMSE"]
    unrestricted_r2_gap = unrestricted_train_r2 - unrestricted_test_r2

    xgb_results_path = REPORTS_DIR / "xgboost_results.json"
    xgb_results = json.loads(xgb_results_path.read_text(encoding="utf-8")) if xgb_results_path.exists() else {}

    rows: list[dict[str, str]] = []

    def row(section: str, kind: str, title: str, text: str, image: str = "") -> None:
        rows.append({"Раздел": section, "Тип": kind, "Заголовок": title, "Текст": text, "Изображение": image})

    pdf_path = Path(os.environ.get("FINAL_REPORT_PDF", REPORTS_DIR / "final_report.pdf"))
    with PdfPages(pdf_path) as pdf:
        page = 1
        page = add_title_page(pdf, page)
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
                "5. Градиентный бустинг: XGBoost",
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
                "Актуальность работы обусловлена ростом объема данных в страховании и необходимостью точнее оценивать индивидуальный риск клиента. Методы машинного обучения позволяют учитывать совместное влияние демографических и поведенческих факторов, благодаря чему расчет страховой стоимости становится более обоснованным.",
                "Цель работы - построить и сопоставить модели регрессии для прогнозирования стоимости медицинской страховки (charges), а также определить признаки, которые сильнее всего влияют на результат.",
                "В качестве факторов рассматриваются возраст клиента (age), индекс массы тела (bmi), количество детей (children), пол (sex), факт курения (smoker) и регион проживания (region). Поэтому работа объединяет две связанные задачи: получение точного прогноза и содержательную интерпретацию факторов страхового риска.",
                "Исследование проводилось последовательно. Сначала были изучены и подготовлены исходные данные, затем построено и оптимизировано одиночное дерево решений, после чего рассмотрены ансамбли Bagging, Random Forest и XGBoost. Разделение на train, validation и test позволило отделить настройку моделей от итоговой проверки и оценить возможное переобучение.",
                "Для сравнения использовалось несколько метрик, поскольку каждая из них отражает отдельную сторону ошибки. Коэффициент детерминации (R2) показывает долю вариации целевой переменной, объясненную моделью. Средняя абсолютная ошибка (MAE) характеризует типичное отклонение прогноза в долларах и сравнительно слабо зависит от единичных крупных промахов.",
                "Среднеквадратическая ошибка (MSE), напротив, сильнее штрафует крупные ошибки, что особенно важно для дорогих страховых случаев. Для интерпретации в исходных единицах вместе с ней использовался корень из среднеквадратической ошибки (RMSE). Средняя абсолютная процентная ошибка (MAPE) дополняет анализ относительной величиной ошибки, однако для небольших значений стоимости (charges) ее следует трактовать осторожно.",
                "Все метрики рассчитывались после обратного преобразования прогнозов в исходную шкалу стоимости. Их сопоставление на train, validation и test использовалось не только для ранжирования моделей, но и для проверки устойчивости полученных результатов.",
            ],
            formulas=[
                r"$MAE=\frac{1}{n}\sum_i |y_i-\hat{y}_i|$",
                r"$MSE=\frac{1}{n}\sum_i (y_i-\hat{y}_i)^2,\quad RMSE=\sqrt{MSE}$",
                r"$MAPE=\frac{100\%}{n}\sum_i\left|\frac{y_i-\hat{y}_i}{y_i}\right|$",
            ],
        )
        row("Введение", "Текст", "Цель работы", "Построить и сравнить модели для прогнозирования стоимости страховки (charges).")

        page = add_text_page(
            pdf,
            page,
            "1. Постановка задачи и описание датасета",
            [
                "В работе решается задача регрессии: по характеристикам клиента требуется предсказать непрерывную величину стоимости медицинской страховки (charges). Исходный датасет включает 1338 наблюдений и 7 столбцов, из которых шесть используются как входные признаки, а один является целевой переменной.",
                "К количественным признакам относятся возраст клиента (age), индекс массы тела (bmi) и количество детей (children). Пол клиента (sex), факт курения (smoker) и регион проживания (region) представлены категориальными признаками. Такое сочетание данных позволяет исследовать как численные зависимости, так и различия между отдельными группами клиентов.",
                f"Перед анализом была проверена целостность набора данных. Пропущенных значений не обнаружено, поэтому этап заполнения не потребовался. Вместе с тем проверка (df.duplicated().sum()) выявила {duplicate_count} полностью повторяющуюся строку. После ее удаления в выборке осталось 1337 уникальных наблюдений.",
            ],
            code="missing_by_column = df.isna().sum()\nduplicate_count = df.duplicated().sum()\n\ndf = df.drop_duplicates().reset_index(drop=True)",
        )
        row("1. Постановка задачи и описание датасета", "Текст", "Описание данных", "Датасет содержит 1338 строк; найден и удален один полный дубликат.")

        page = add_image_page(
            pdf,
            page,
            "1. Постановка задачи и описание датасета",
            figure_paths["raw_preview"],
            "Рисунок 1 - Фрагмент исходного датасета. Таблица показывает исходные количественные и категориальные признаки до кодирования, а также целевую переменную (charges), выраженную в долларах.",
        )

        page = add_image_page(
            pdf,
            page,
            "1. Постановка задачи и описание датасета",
            figure_paths["charges_stats"],
            "Рисунок 2 - Описательные статистики целевой переменной (charges). Максимальное значение заметно превышает среднее, а стандартное отклонение сопоставимо со средним, поэтому распределение требует отдельного анализа формы и выбросов.",
        )
        row("1. Постановка задачи и описание датасета", "Рисунок", "Статистики charges", "Добавлена таблица статистик целевой переменной (charges).", str(figure_paths["charges_stats"]))

        page = add_text_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            [
                "Разведочный анализ начался с изучения распределения стоимости страховки (charges). Основная часть наблюдений сосредоточена в нижнем ценовом диапазоне, тогда как небольшая группа дорогих случаев образует длинный правый хвост.",
                f"Коэффициент асимметрии исходной целевой переменной равен {target_skew:.2f}, что количественно подтверждает правосторонний перекос. По этой причине высокие значения были рассмотрены отдельно: прежде чем считать их выбросами, необходимо было проверить, связаны ли они с реальными характеристиками клиентов.",
            ],
        )
        row("2. Разведочный анализ и предобработка данных", "Текст", "Анализ выбросов", f"Выше $35 000 находятся {outlier_count} объектов; {outlier_smoker_share:.1f}% из них - курильщики.")

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["charges_distribution"],
            "Рисунок 3 - Распределение целевой переменной (charges). Среднее расположено правее медианы, а частоты постепенно снижаются вдоль длинного правого хвоста; это подтверждает правостороннюю асимметрию и мотивирует логарифмирование таргета.",
        )

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["charges_boxplot_horizontal"],
            "Рисунок 4 - Горизонтальный boxplot целевой переменной (charges). Высокие значения лежат за верхней границей диаграммы, однако сам по себе boxplot еще не доказывает, что они являются ошибками: для этого необходимо проверить их связь с признаками клиентов.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Boxplot charges", "Горизонтальный boxplot делает выбросы по (charges) более читаемыми.", str(figure_paths["charges_boxplot_horizontal"]))

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["charges_boxplot_by_smoker"],
            "Рисунок 5 - Стоимость страховки (charges) в группах по признаку (smoker). Анализ показал, что 133 объекта, или 9,9% выборки, имеют стоимость выше 35 000 долларов, причем 97,7% этих объектов являются курильщиками. Следовательно, большинство высоких значений отражает реальную закономерность, а не ошибочные выбросы; удалять их из выборки не следует.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Boxplot charges по smoker", "Показано, что высокие значения (charges) в основном связаны с курящими клиентами.", str(figure_paths["charges_boxplot_by_smoker"]))

        page = add_text_page(
            pdf,
            page,
            "Логарифмирование целевой переменной",
            [
                "Выявленная правосторонняя асимметрия послужила основанием для логарифмирования целевой переменной (charges). Модели обучались на величине log(charges + 1), поэтому редкие дорогие полисы меньше доминировали в функции потерь, а различия в основной части распределения становились заметнее.",
                f"После преобразования коэффициент асимметрии снизился с {target_skew:.2f} до {log_target_skew:.2f}, и распределение стало существенно ближе к симметричному. Полученные моделью прогнозы затем возвращались в исходную шкалу с помощью обратного преобразования, поэтому все итоговые метрики выражены в реальной стоимости страховки.",
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
            "Рисунок 6 - Корреляции количественных признаков до и после логарифмирования таргета. Значимых корреляций между самими признаками не обнаружено. Наиболее сильная линейная связь наблюдается между возрастом клиента (age) и логарифмом стоимости: коэффициент возрастает примерно с 0,30 до 0,53. Связь с индексом массы тела (bmi) и количеством детей (children) остается слабой, поэтому их влияние может быть нелинейным и требует проверки моделями деревьев.",
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Сравнение корреляционных матриц", "Добавлено сравнение корреляций с обычным таргетом (charges) и логарифмированным таргетом log(charges + 1).", str(figure_paths["correlation_original_vs_log"]))

        page = add_text_page(
            pdf,
            page,
            "Кодирование и разделение данных",
            [
                "После анализа распределений данные были подготовлены к моделированию. Категориальные признаки (sex), (smoker) и (region) преобразованы с помощью OneHotEncoder. Кодировщик обучался только на тренировочной части, что исключает перенос информации из validation и test в процесс подготовки признаков.",
                "При разделении учитывались правосторонний перекос стоимости и связь дорогих случаев с курением. Обычная стратификация для непрерывной целевой переменной невозможна, поэтому значения (charges) были разбиты на пять квантильных интервалов, а номер интервала объединен с категорией (smoker). Полученная комбинированная метка одновременно сохраняет долю курильщиков и представительство разных диапазонов стоимости.",
                f"Сначала 60% наблюдений были отнесены к train, после чего оставшиеся 40% поровну разделены на validation и test с повторной стратификацией. В результате получено {len(train)} объектов в train, {len(val)} объектов в validation и {len(test)} объектов в test. После OneHotEncoding сформировано {train.drop(columns=['charges', 'charges_original']).shape[1]} входных признаков.",
            ],
            code="df['charges_bin'] = pd.qcut(df['charges'], q=5, duplicates='drop')\ndf['strata'] = df['charges_bin'].astype(str) + '_' + df['smoker']\n\ntrain_df, temp_df = train_test_split(\n    df, test_size=0.40, random_state=42, stratify=df['strata']\n)\nvalidation_df, test_df = train_test_split(\n    temp_df, test_size=0.50, random_state=42, stratify=temp_df['strata']\n)",
        )
        row("2. Разведочный анализ и предобработка данных", "Код", "Стратифицированное разделение", "Использована комбинированная стратификация через интервалы (charges) и признак (smoker).")

        page = add_image_page(
            pdf,
            page,
            "2. Разведочный анализ и предобработка данных",
            figure_paths["processed_preview"],
            "Рисунок 7 - Фрагмент датасета после предобработки. Количественные признаки сохранены, категории преобразованы в бинарные столбцы, а поле (charges) содержит логарифмированный таргет. Исходная стоимость хранится в (charges_original) и используется для обратного преобразования и расчета метрик.",
            after_body=[
                "Таким образом, предобработка привела данные к единому числовому представлению без потери содержательно важных дорогих случаев. После удаления одного дубликата осталось 1337 наблюдений и 8 входных признаков.",
                "Комбинированная стратификация обеспечила близкую структуру train, validation и test, поэтому полученные выборки можно последовательно использовать для обучения, настройки и итогового сравнения моделей.",
            ],
        )
        row("2. Разведочный анализ и предобработка данных", "Рисунок", "Итоговый датасет", "Показан фрагмент данных после OneHotEncoding и логарифмирования.", str(figure_paths["processed_preview"]))

        page = add_text_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            [
                "Первой моделью стало дерево решений, которое формирует прогноз с помощью последовательных разбиений пространства признаков. В каждом узле выбирается локально лучшее условие по критерию ошибки. Благодаря этому структуру дерева легко объяснить, однако при отсутствии ограничений оно может создавать листья для отдельных наблюдений и фактически запоминать train.",
                f"Этот риск подтвердился на практике: неограниченное дерево достигло глубины {unrestricted_tree.get_depth()} и получило на train R2 = {unrestricted_train_r2:.4f} при RMSE = {money(unrestricted_train_rmse)}. На test коэффициент R2 снизился до {unrestricted_test_r2:.4f}, а RMSE вырос до {money(unrestricted_test_rmse)}. Разрыв R2, равный {unrestricted_r2_gap:.4f}, указывает на выраженное переобучение.",
                "Такой результат соответствует компромиссу между смещением и разбросом: сложное дерево имеет низкое смещение, но становится чувствительным к конкретному составу тренировочной выборки. Поэтому дальнейшая настройка была направлена на ограничение глубины и размеров листьев, чтобы уменьшить разброс при сохранении основной зависимости.",
            ],
            formulas=[
                r"$\mathrm{Ошибка}=\mathrm{Bias}^2+\mathrm{Variance}+\mathrm{Noise}$",
            ],
        )
        row("3. Базовая модель и проблема переобучения", "Текст", "Неограниченное дерево", f"Глубина {unrestricted_tree.get_depth()}, разрыв train-test R2={unrestricted_r2_gap:.4f}.")

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            REPORTS_DIR / "figures" / "decision_tree_structure_first_3_levels.png",
            "Рисунок 8 - Первые три уровня оптимизированного дерева решений. В каждом узле показано условие разбиения, а движение по ветвям последовательно сужает группу объектов; полное дерево глубже, поэтому на рисунке оставлены первые уровни для читаемости.",
        )

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            figure_paths["tree_overfitting"],
            "Рисунок 9 - Метрики дерева до и после оптимизации. Ограничения почти не ухудшили качество на test, но резко сократили разрыв с train: это показывает, что регуляризация убрала запоминание отдельных наблюдений и повысила обобщающую способность.",
        )

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            REPORTS_DIR / "figures" / "decision_tree_best_params.png",
            "Рисунок 10 - Лучшие параметры оптимизированного дерева решений. После двухэтапного поиска итоговая глубина составила 10; значения (min_samples_split), (min_samples_leaf) и (ccp_alpha) ограничивают дробление узлов и снижают разброс модели.",
        )
        row("3. Базовая модель и проблема переобучения", "Рисунок", "Параметры дерева", "Лучшие параметры оптимизированного дерева решений.", "reports/figures/decision_tree_best_params.png")

        page = add_text_page(
            pdf,
            page,
            "Оптимизация гиперпараметров дерева",
            [
                "Гиперпараметры дерева подбирались с помощью GridSearchCV и пятиблочной кросс-валидации KFold с перемешиванием. Сначала широкая сетка определила перспективную область, после чего значения глубины и размеров узлов проверялись с шагом 1 рядом с найденным оптимумом. Выборка validation в поиске не участвовала и служила независимой проверкой выбранной конфигурации.",
                f"В результате получено оптимизированное дерево глубины {tree_depth}. {model_metrics_sentence(metrics_df, 'Оптимизированное дерево')} Разрыв R2 между train и test сократился до {tree_r2_gap:.4f}. Следовательно, ограничения действительно уменьшили переобучение, не разрушив полезную структуру модели.",
            ],
            code="coarse_grid = {\n    'max_depth': [2, 4, 6, 8, 10, 12, 16, None],\n    'min_samples_split': [2, 10, 20, 40],\n    'min_samples_leaf': [1, 4, 8, 16],\n    'criterion': ['squared_error', 'absolute_error'],\n    'ccp_alpha': [0.0, 0.0001]\n}\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\ngrid = GridSearchCV(tree, coarse_grid, cv=cv, scoring=r2_original_scorer)",
        )

        page = add_image_page(
            pdf,
            page,
            "3. Базовая модель и проблема переобучения",
            REPORTS_DIR / "figures" / "decision_tree_residuals_test.png",
            "Рисунок 11 - Остатки оптимизированного дерева на test. Большинство точек находится около нулевой линии, поэтому модель в целом улавливает основную зависимость. Однако разброс увеличивается для дорогих полисов, а полосы точек возникают потому, что объекты в одном листе получают одинаковый прогноз; качество приемлемое, но редкие крупные ошибки сохраняются.",
            after_body=[
                f"Таким образом, двухэтапная настройка заменила переобученное дерево глубины {unrestricted_tree.get_depth()} на дерево глубины {tree_depth} с близкими значениями R2 на train ({tree_train_r2:.4f}), validation ({tree_metrics.loc['Validation', 'R2']:.4f}) и test ({tree_test_r2:.4f}).",
                "Модель сохранила интерпретируемость, но структура остатков показала чувствительность к дорогим страховым случаям. Это стало основанием перейти к ансамблям, способным снизить разброс одиночного дерева.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            [
                "Результаты одиночного дерева показали, что основной резерв улучшения связан со снижением разброса. Поэтому следующим шагом стали ансамблевые модели, объединяющие прогнозы нескольких деревьев. Для сопоставимости использовались те же признаки, логарифмированная целевая переменная и неизменное разбиение данных.",
                "В Bagging каждое дерево обучается на собственной bootstrap-подвыборке, после чего прогнозы усредняются. Random Forest дополнительно ограничивает набор признаков, доступных при поиске разбиения. Благодаря этому деревья становятся менее похожими друг на друга, а их совместный прогноз - более устойчивым.",
            ],
        )
        row("4. Ансамблевые модели", "Текст", "Bagging и Random Forest", "Ансамбли использовались для снижения разброса одиночного дерева.")

        page = add_text_page(
            pdf,
            page,
            "4.1 Оптимизация Bagging",
            [
                "Для Bagging применялся двухэтапный GridSearchCV. На первом этапе подбирались число деревьев (n_estimators), доля объектов (max_samples), доля признаков (max_features), а также глубина и минимальные размеры узлов базового дерева. Затем вокруг лучших значений была построена уточненная сетка.",
                model_metrics_sentence(metrics_df, "Bagging") + " Разрыв между train и test остается умеренным, поэтому усреднение деревьев действительно контролирует переобучение. Более высокое значение на validation можно объяснить случайными различиями между конечными подвыборками.",
            ],
            code="bagging_grid = {\n    'n_estimators': [50, 100, 200],\n    'max_samples': [0.6, 0.8, 1.0],\n    'max_features': [0.6, 0.8, 1.0],\n    'estimator__max_depth': [4, 6, 8, 10, None],\n    'estimator__min_samples_split': [2, 10, 20],\n    'estimator__min_samples_leaf': [1, 4, 8]\n}",
        )

        page = add_image_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            REPORTS_DIR / "figures" / "bagging_best_params.png",
            "Рисунок 12 - Лучшие параметры Bagging. Сочетание параметров определяет разнообразие базовых деревьев и объем данных для каждого из них; итоговая конфигурация улучшила R2 и RMSE относительно оптимизированного одиночного дерева.",
        )

        page = add_text_page(
            pdf,
            page,
            "4.2 Оптимизация Random Forest",
            [
                "Random Forest оптимизировался по той же двухэтапной схеме: широкий GridSearchCV определял перспективную область, а уточненный поиск проверял соседние значения. Помимо числа и глубины деревьев, параметр (max_features) регулировал случайную долю признаков, доступных при разбиении, а (bootstrap) включал выбор объектов с возвращением.",
                model_metrics_sentence(metrics_df, "Random Forest") + " На train лес превосходит Bagging заметнее, чем на test. Следовательно, дополнительная случайность признаков дает лишь небольшой прирост обобщающего качества на рассматриваемых данных.",
            ],
            code="rf_grid = {\n    'n_estimators': [100, 200],\n    'max_depth': [4, 6, 8, 10, 12, None],\n    'min_samples_split': [2, 10, 20],\n    'min_samples_leaf': [1, 4, 8],\n    'max_features': ['sqrt', 0.7, 1.0],\n    'bootstrap': [True],\n    'ccp_alpha': [0.0, 0.0001]\n}",
        )

        page = add_image_page(
            pdf,
            page,
            "4. Ансамблевые модели: Bagging и Random Forest",
            REPORTS_DIR / "figures" / "random_forest_best_params.png",
            "Рисунок 13 - Лучшие параметры Random Forest. Подбор (max_features) управляет декорреляцией деревьев: чем меньше доступно признаков, тем разнообразнее деревья, но слишком сильное ограничение может ухудшить отдельные разбиения.",
        )

        page = add_text_page(
            pdf,
            page,
            "Оценка ансамблей с помощью OOB-score",
            [
                "Для bootstrap-ансамблей качество дополнительно оценивалось с помощью OOB-score. В каждой bootstrap-подвыборке часть объектов отсутствует, поэтому их можно предсказать только теми деревьями, которые не использовали эти наблюдения при обучении. Так формируется внутренняя оценка качества без выделения еще одной выборки.",
                "Значения OOB сопоставлялись с пятиблочной кросс-валидацией и результатом на test. Для одиночного дерева такая оценка не определена, поскольку оно не является bootstrap-композицией. У Bagging и Random Forest OOB оказался немного выше KFold, однако близость оценок сохраняет общий вывод об устойчивости ансамблей.",
            ],
            code="bagging_oob = BaggingRegressor(\n    estimator=best_tree,\n    n_estimators=best_n,\n    bootstrap=True,\n    oob_score=True,\n    random_state=42\n)\n\nrf_oob = RandomForestRegressor(\n    n_estimators=best_n,\n    bootstrap=True,\n    oob_score=True,\n    random_state=42\n)",
        )

        page = add_image_page(
            pdf,
            page,
            "4.3 Сравнение OOB-score и KFold",
            figure_paths["oob_comparison"],
            "Рисунок 14 - OOB, KFold и test для дерева и ансамблей. Для Bagging OOB R2 равен 0,8628 против 0,8478 по KFold, для Random Forest - 0,8637 против 0,8466; тестовые значения 0,8475 и 0,8485 близки к KFold, поэтому вывод о качестве не зависит от одной процедуры валидации.",
        )

        page = add_image_page(
            pdf,
            page,
            "4.4 Важность признаков Random Forest",
            REPORTS_DIR / "figures" / "random_forest_feature_importance.png",
            "Рисунок 15 - Важность признаков Random Forest. Наибольший вклад имеют факт курения (smoker_yes) и возраст (age), что согласуется с EDA; индекс массы тела (bmi) занимает третье место, а пол и регион влияют слабо.",
            after_body=[
                "В целом Bagging и Random Forest улучшили R2 и RMSE относительно оптимизированного дерева, сохранив близкие результаты на train, validation и test. Согласованность OOB и KFold дополнительно подтверждает, что вывод не зависит от единственного способа валидации.",
                "Random Forest получил небольшое преимущество благодаря декорреляции деревьев, но разница с Bagging невелика. Сильное влияние признака курения (smoker) приводит модели к похожим ключевым разбиениям даже при различиях в механизме построения ансамбля.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "5. Градиентный бустинг: XGBoost",
            [
                "После параллельных ансамблей был рассмотрен градиентный бустинг XGBoost, реализованный моделью XGBRegressor. В отличие от Bagging и Random Forest, он строит деревья последовательно: каждое новое дерево направлено на исправление ошибок уже сформированного ансамбля.",
                "На очередном шаге вычисляются псевдоостатки, задающие направление уменьшения функции потерь. Для квадратичной ошибки они совпадают с обычными остатками (y - F). Новое дерево приближает эти значения, а величина его вклада регулируется скоростью обучения (learning_rate).",
                "В XGBoost последовательное уменьшение смещения дополняется регуляризацией и подвыборкой объектов и признаков. Такое сочетание позволяет наращивать качество постепенно, не делая каждое отдельное дерево чрезмерно сложным.",
            ],
            formulas=[
                r"$r_i^{(m)}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}$",
                r"$F_m(x)=F_{m-1}(x)+\eta h_m(x)$",
            ],
        )
        row("5. Градиентный бустинг: XGBoost", "Текст", "Двухэтапный подбор", "Сначала широкий RandomizedSearchCV, затем уточнение вокруг лучших параметров.")

        page = add_text_page(
            pdf,
            page,
            "5.1 Оптимизация XGBoost",
            [
                f"Поиск гиперпараметров также проводился в два этапа. Сначала RandomizedSearchCV проверил 60 комбинаций на пяти блоках KFold, затем еще 50 комбинаций были исследованы в окрестности лучшего решения. Средний CV R2 вырос с {xgb_results.get('coarse_best_score', float('nan')):.4f} до {xgb_results.get('refined_best_score', float('nan')):.4f}, а наилучшее качество на validation было достигнуто примерно после {xgb_results.get('best_stage', 'NA')} деревьев.",
                model_metrics_sentence(metrics_df, "XGBoost") + " Близкие значения на train, validation и test показывают, что регуляризация и подвыборки удержали переобучение, несмотря на большое число последовательных итераций.",
            ],
            code="xgb_grid = {\n    'n_estimators': [150, 250, 400, 600, 800],\n    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08],\n    'max_depth': [2, 3, 4, 5],\n    'min_child_weight': [1, 3, 5, 8],\n    'subsample': [0.7, 0.8, 0.9, 1.0],\n    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],\n    'reg_lambda': [0.5, 1, 3, 5, 10],\n    'reg_alpha': [0, 0.01, 0.1, 0.5]\n}",
        )

        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг: XGBoost",
            figure_paths["xgboost_best_params"],
            "Рисунок 16 - Лучшие параметры XGBoost после уточненного поиска. Небольшая скорость обучения (learning_rate = 0,01), глубина 3 и регуляризация дают постепенное обучение без чрезмерно сложных отдельных деревьев.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг: XGBoost",
            REPORTS_DIR / "figures" / "xgboost_learning_curve.png",
            "Рисунок 17 - Кривая качества XGBoost по числу деревьев. Train продолжает улучшаться, но validation выходит на плато примерно к 648 деревьям; выбранная остановка сохраняет максимальное обобщающее качество и не расходует итерации на запоминание train.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг: XGBoost",
            REPORTS_DIR / "figures" / "xgboost_residuals_test.png",
            "Рисунок 18 - Остатки XGBoost на тестовой выборке. Облако в основном центрировано около нуля, поэтому систематического общего смещения не видно и модель можно считать адекватной. Вместе с тем разброс растет у дорогих полисов, следовательно, качество хорошее, но неоднородность ошибок полностью не устранена.",
        )
        page = add_image_page(
            pdf,
            page,
            "5. Градиентный бустинг: XGBoost",
            REPORTS_DIR / "figures" / "xgboost_feature_importance.png",
            "Рисунок 19 - Важность признаков XGBoost. Признак курения (smoker_yes) занимает около 70% суммарной важности, затем следуют возраст (age), количество детей (children) и индекс массы тела (bmi). Это подтверждает гипотезу EDA о сильном влиянии курения.",
            after_body=[
                f"На test модель XGBoost достигла R2 = {metrics_df[(metrics_df['Модель'] == 'XGBoost') & (metrics_df['Выборка'] == 'Test')]['R2'].iloc[0]:.4f} и RMSE = {money(metrics_df[(metrics_df['Модель'] == 'XGBoost') & (metrics_df['Выборка'] == 'Test')]['RMSE'].iloc[0])}, показав лучший результат по этим двум метрикам.",
                "Кривая обучения и небольшой разрыв между выборками не указывают на выраженное переобучение. При этом остатки показывают, что наиболее дорогие полисы по-прежнему остаются самой сложной областью прогноза.",
            ],
        )

        page = add_text_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            [
                "На заключительном этапе результаты всех моделей были сведены в единую систему сравнения. Сопоставление метрик на train и test позволило одновременно оценить точность прогноза и возможное переобучение, а результаты на validation показали устойчивость конфигураций, выбранных во время настройки.",
                "Выборка test не участвовала ни в обучении, ни в подборе гиперпараметров, поэтому именно она использовалась для итоговой оценки обобщающей способности. Численные метрики были дополнены анализом остатков, диаграммами фактических и предсказанных значений, важностью признаков, SHAP-интерпретацией и измерением скорости инференса.",
            ],
        )

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            figure_paths["split_metrics"],
            "Рисунок 20 - Итоговые метрики на train, validation и test. Все показатели рассчитаны в исходной шкале после преобразования exp(prediction) - 1; таблица позволяет одновременно сравнить качество и разрыв между выборками для каждой модели.",
        )
        row("6. Сравнительный анализ моделей", "Рисунок", "Метрики по выборкам", "Добавлена итоговая таблица train, validation и test.", str(figure_paths["split_metrics"]))

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            figure_paths["r2_train_test"],
            "Рисунок 21 - Сравнение моделей по R2 на train и test. Числа над столбцами показывают, что у всех оптимизированных моделей разрыв невелик; наибольший разрыв среди них наблюдается у Random Forest, а XGBoost показывает лучший test R2 при умеренном отличии от train.",
        )
        row("6. Сравнительный анализ моделей", "Рисунок", "R2 train/test", "Добавлены разные цвета для train и test.", str(figure_paths["r2_train_test"]))

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_errors_models.png",
            "Рисунок 22 - MAE и RMSE моделей на test. RMSE находится в узком диапазоне от 4 300 до 4 565 долларов: все модели используют одни признаки, обучаются на логарифмированном таргете и сталкиваются с одинаковыми редкими дорогими случаями, которые доминируют в квадрате ошибки. У дерева MAE ниже, но RMSE выше: большинство его прогнозов близки к факту, однако отдельные крупные промахи сильнее ухудшают RMSE.",
        )

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_actual_vs_predicted_all_models.png",
            "Рисунок 23 - Предсказанные и фактические значения для всех моделей. Чем ближе точки к диагонали, тем точнее прогноз; XGBoost формирует наиболее компактное облако, но в верхнем диапазоне стоимости все модели чаще отклоняются от идеальной линии.",
        )

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_residuals_all_models.png",
            "Рисунок 24 - Остатки всех моделей на test. У ансамблей облака более сглажены, чем у одиночного дерева, а выраженного постоянного смещения относительно нуля нет. Расширение облака справа у всех моделей подтверждает систематически более сложный прогноз дорогих полисов.",
        )
        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_feature_importance_all_models.png",
            "Рисунок 25 - Сравнение важности признаков. Во всех ансамблях главным фактором остается курение (smoker_yes), далее обычно идут возраст (age) и индекс массы тела (bmi). Признаки региона и пола практически не влияют на целевую переменную, что подтверждает предварительный анализ.",
        )
        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_shap_summary_random_forest.png",
            "Рисунок 26 - SHAP-анализ Random Forest. Точки показывают направление и величину вклада признака в прогноз log(charges + 1): курение и больший возраст чаще сдвигают прогноз вверх. SHAP дополняет встроенную важность тем, что показывает не только силу, но и направление влияния.",
        )

        page = add_image_page(
            pdf,
            page,
            "6. Сравнительный анализ моделей",
            REPORTS_DIR / "figures" / "comparison_inference_speed.png",
            "Рисунок 27 - Среднее время инференса моделей. Одиночное дерево прогнозирует быстрее всего, XGBoost занимает промежуточное положение, а Bagging и Random Forest медленнее из-за вычисления прогнозов множества независимых деревьев; для данного небольшого датасета все значения остаются практически приемлемыми.",
        )

        test_metrics = metrics_df[metrics_df["Выборка"] == "Test"].set_index("Модель")
        best_test = test_metrics.sort_values("R2", ascending=False).iloc[0]
        best_model_name = test_metrics["R2"].idxmax()
        best_by_split = metrics_df[metrics_df["Модель"] == best_model_name].set_index("Выборка")
        tree_test = test_metrics.loc["Оптимизированное дерево"]
        bagging_test = test_metrics.loc["Bagging"]
        forest_test = test_metrics.loc["Random Forest"]
        page = add_text_page(
            pdf,
            page,
            "Заключение",
            [
                f"В ходе работы был исследован набор из 1338 наблюдений для прогнозирования стоимости медицинской страховки (charges). Пропущенных значений не обнаружено, одна полностью повторяющаяся строка была удалена. Категориальные признаки преобразованы методом OneHotEncoding, а данные разделены на train, validation и test с комбинированной стратификацией по интервалам стоимости и признаку курения (smoker).",
                f"Разведочный анализ выявил правостороннюю асимметрию целевой переменной: коэффициент асимметрии исходного распределения составил {target_skew:.2f}. Среди {outlier_count} полисов дороже 35 000 долларов доля курильщиков достигла {outlier_smoker_share:.1f}%, поэтому высокие значения были признаны содержательно объяснимыми и не удалялись. Логарифмирование уменьшило асимметрию до {log_target_skew:.2f} и позволило обучать модели в более устойчивой шкале.",
                f"На test оптимизированное дерево получило R2 = {tree_test['R2']:.4f}, Bagging - {bagging_test['R2']:.4f}, Random Forest - {forest_test['R2']:.4f}, а {best_model_name} - {best_test['R2']:.4f}. Лучшей по R2 и RMSE стала модель {best_model_name}: RMSE составил {money(best_test['RMSE'])}, MAE - {money(best_test['MAE'])}. При этом оптимизированное дерево показало меньшие MAE ({money(tree_test['MAE'])}) и MAPE ({tree_test['MAPE']:.2f}%), поэтому преимущество XGBoost относится прежде всего к объясненной вариации и снижению крупных ошибок.",
                f"Для {best_model_name} значения R2 на train, validation и test равны соответственно {best_by_split.loc['Train', 'R2']:.4f}, {best_by_split.loc['Validation', 'R2']:.4f} и {best_by_split.loc['Test', 'R2']:.4f}. Небольшой разрыв между выборками не указывает на выраженное переобучение, а test R2 означает, что модель объясняет около {best_test['R2'] * 100:.1f}% вариации стоимости на ранее не использованных данных. Основная часть оставшейся ошибки связана с наиболее дорогими полисами, где разброс остатков возрастает.",
                "Интерпретация лучшей модели согласуется с выводами EDA. Признак курения (smoker_yes) формирует около 70% встроенной важности XGBoost, что подтверждает обнаруженную на этапе анализа концентрацию дорогих полисов среди курильщиков. Возраст (age) также остается значимым фактором и соответствует наиболее сильной корреляции с log(charges + 1). Индекс массы тела (bmi) и количество детей (children) дают меньший, преимущественно нелинейный вклад, тогда как пол (sex) и регион (region) почти не влияют на прогноз.",
                "Дальнейшее улучшение качества целесообразно начать с расширения данных: добавить сведения о хронических заболеваниях, страховом плане, уровне дохода и истории обращений, а затем проверить модель на внешней или более поздней выборке. В рамках текущего датасета можно исследовать взаимодействия возраста, курения и индекса массы тела, отдельно оптимизировать ошибку для дорогих полисов, применить повторную кросс-валидацию и построить интервальные прогнозы. Это позволит не только повысить среднее качество, но и оценивать неопределенность в наиболее рискованных случаях.",
                f"Итоговой моделью по совокупности R2 и RMSE выбран {best_model_name}. Оптимизированное дерево остается наиболее простым и быстрым решением, Random Forest дает близкое качество и удобен для OOB- и SHAP-анализа, однако XGBoost лучше всего сокращает крупные ошибки и обеспечивает наиболее высокий test R2.",
            ],
        )
        row("Заключение", "Текст", "Итоговый вывод", f"Лучшая модель по test R2: {best_model_name}.")

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

    build_csv(rows)
    print(f"PDF saved: {pdf_path}")
    print(f"CSV saved: {REPORTS_DIR / 'final_report.csv'}")


if __name__ == "__main__":
    main()
