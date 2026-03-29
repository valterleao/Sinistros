from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "CSV"
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 42
MAX_MODEL_ROWS = 300_000
FIGURE_FORMAT = "svg"

REQUIRED_COLUMNS = [
    "data_inversa",
    "dia_semana",
    "horario",
    "uf",
    "br",
    "km",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "veiculos",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
]

LEAKAGE_COLUMNS = [
    "classificacao_acidente",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "pessoas",
]

CATEGORICAL_FEATURES = [
    "uf",
    "br",
    "dia_semana",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]

NUMERIC_FEATURES = [
    "mes",
    "hora",
    "km",
    "veiculos",
    "fim_de_semana",
]


@dataclass
class AnalysisArtifacts:
    metrics: dict
    class_distribution: dict
    top_feature_importances: list[dict]
    grouped_feature_importances: list[dict]
    top_risk_factors: list[dict]
    sampled_rows: int
    total_rows: int
    dropped_rows: int


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text)


def parse_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
    has_dot = text.str.contains(r"\.", na=False)
    has_comma = text.str.contains(",", na=False)
    both_mask = has_dot & has_comma
    comma_only_mask = has_comma & ~has_dot

    text.loc[both_mask] = (
        text.loc[both_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    text.loc[comma_only_mask] = text.loc[comma_only_mask].str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def discover_csv_files(csv_dir: Path) -> list[Path]:
    files = sorted(csv_dir.glob("datatran*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV DATATRAN encontrado em {csv_dir}")
    return files


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    header = pd.read_csv(file_path, sep=";", nrows=0, encoding="latin1")
    available_columns = [column for column in REQUIRED_COLUMNS if column in header.columns]
    if "classificacao_acidente" not in available_columns:
        raise ValueError(f"Arquivo sem classificacao_acidente: {file_path.name}")

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin1",
        usecols=available_columns,
        low_memory=False,
    )

    year_match = re.search(r"(\d{4})", file_path.stem)
    df["ano_arquivo"] = int(year_match.group(1)) if year_match else pd.NA
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    classificacao = df["classificacao_acidente"].map(normalize_text)
    mortos = parse_numeric(df["mortos"]).fillna(0)
    feridos = parse_numeric(df["feridos"]).fillna(0)
    feridos_leves = parse_numeric(df["feridos_leves"]).fillna(0)
    feridos_graves = parse_numeric(df["feridos_graves"]).fillna(0)
    total_vitimas = mortos + feridos + feridos_leves + feridos_graves

    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target[classificacao.str.contains("fatal|ferid", regex=True, na=False)] = 1
    target[classificacao.str.contains("sem", regex=True, na=False)] = 0
    target[total_vitimas > 0] = 1
    target[(total_vitimas == 0) & target.isna()] = 0
    return target


def prepare_dataset(files: Iterable[Path]) -> tuple[pd.DataFrame, pd.Series, dict]:
    frames: list[pd.DataFrame] = []
    for file_path in files:
        frames.append(read_csv_safely(file_path))

    df = pd.concat(frames, ignore_index=True)
    df["com_vitima"] = build_target(df)

    df["data_dt"] = pd.to_datetime(df["data_inversa"], dayfirst=True, errors="coerce")
    missing_date_mask = df["data_dt"].isna()
    if missing_date_mask.any():
        df.loc[missing_date_mask, "data_dt"] = pd.to_datetime(
            df.loc[missing_date_mask, "data_inversa"],
            format="%Y-%m-%d",
            errors="coerce",
        )

    df["hora"] = pd.to_numeric(df["horario"].astype(str).str.slice(0, 2), errors="coerce")
    df["mes"] = df["data_dt"].dt.month
    df["fim_de_semana"] = df["data_dt"].dt.dayofweek.isin([5, 6]).astype(int)
    df["km"] = parse_numeric(df["km"])
    df["veiculos"] = parse_numeric(df["veiculos"])
    df["br"] = df["br"].astype(str).str.strip().replace({"nan": "Nao informado"})

    for column in CATEGORICAL_FEATURES:
        if column == "br":
            continue
        df[column] = df[column].astype(str).str.strip().replace(
            {"nan": "Nao informado", "": "Nao informado", "NA": "Nao informado"}
        )

    before_drop = len(df)
    df = df.dropna(subset=["com_vitima"])
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df["com_vitima"].astype(int)

    summary = {
        "total_rows": int(before_drop),
        "usable_rows": int(len(df)),
        "dropped_rows": int(before_drop - len(df)),
        "class_distribution": {
            "sem_vitima": int((y == 0).sum()),
            "com_vitima": int((y == 1).sum()),
        },
    }
    return X, y, summary


def stratified_sample(X: pd.DataFrame, y: pd.Series, max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y

    X_sampled, _, y_sampled, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_sampled, y_sampled


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        min_samples_split=200,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def export_confusion_matrix(matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusao - Arvore de Decisao")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"matriz_confusao.{FIGURE_FORMAT}", format=FIGURE_FORMAT)
    plt.close()


def export_feature_importances(importances: pd.DataFrame) -> None:
    top_features = importances.head(15).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["atributo"], top_features["importancia"], color="#1f77b4")
    plt.title("15 atributos mais importantes")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"importancia_atributos.{FIGURE_FORMAT}", format=FIGURE_FORMAT)
    plt.close()


def export_tree_preview(pipeline: Pipeline, feature_names: list[str]) -> None:
    plt.figure(figsize=(24, 12))
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["sem_vitima", "com_vitima"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"arvore_preview.{FIGURE_FORMAT}", format=FIGURE_FORMAT)
    plt.close()


def aggregate_feature_importances(importances: pd.DataFrame) -> pd.DataFrame:
    original_features = sorted(CATEGORICAL_FEATURES + NUMERIC_FEATURES, key=len, reverse=True)
    grouped_rows = []

    for _, row in importances.iterrows():
        attribute = row["atributo"]
        if attribute.startswith("categorical__"):
            remainder = attribute.replace("categorical__", "", 1)
        elif attribute.startswith("numeric__"):
            remainder = attribute.replace("numeric__", "", 1)
        else:
            remainder = attribute

        original_name = remainder
        for feature in original_features:
            if remainder == feature or remainder.startswith(f"{feature}_"):
                original_name = feature
                break

        grouped_rows.append({"variavel": original_name, "importancia": row["importancia"]})

    return (
        pd.DataFrame(grouped_rows)
        .groupby("variavel", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
    )


def calculate_top_risk_factors(df_sampled: pd.DataFrame, y_sampled: pd.Series) -> list[dict]:
    analysis_df = df_sampled.copy()
    analysis_df["com_vitima"] = y_sampled.values

    result = (
        analysis_df.groupby("causa_acidente", dropna=False)["com_vitima"]
        .agg(["mean", "count"])
        .query("count >= 500")
        .sort_values(["mean", "count"], ascending=[False, False])
        .head(10)
        .reset_index()
    )
    result["mean"] = result["mean"].round(4)
    return result.rename(
        columns={
            "causa_acidente": "categoria",
            "mean": "taxa_com_vitima",
            "count": "quantidade",
        }
    ).to_dict(orient="records")


def run_analysis() -> AnalysisArtifacts:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for legacy_png in OUTPUT_DIR.glob("*.png"):
        legacy_png.unlink(missing_ok=True)
    files = discover_csv_files(CSV_DIR)
    X, y, summary = prepare_dataset(files)
    X_model, y_model = stratified_sample(X, y, MAX_MODEL_ROWS)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y_model,
        test_size=0.2,
        stratify=y_model,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["sem_vitima", "com_vitima"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
        cm,
        index=["Real sem_vitima", "Real com_vitima"],
        columns=["Previsto sem_vitima", "Previsto com_vitima"],
    )
    cm_df.to_csv(OUTPUT_DIR / "matriz_confusao.csv", index=True)
    export_confusion_matrix(cm_df)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importances = pd.DataFrame(
        {
            "atributo": feature_names,
            "importancia": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importancia", ascending=False)
    feature_importances.to_csv(OUTPUT_DIR / "importancia_atributos.csv", index=False)
    grouped_importances = aggregate_feature_importances(feature_importances)
    grouped_importances.to_csv(OUTPUT_DIR / "importancia_por_variavel.csv", index=False)
    export_feature_importances(feature_importances)
    export_tree_preview(pipeline, list(feature_names))

    top_risk_factors = calculate_top_risk_factors(X_model, y_model)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "classification_report": report,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    with open(OUTPUT_DIR / "metricas_modelo.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    dataset_summary = {
        "arquivos_processados": [file.name for file in files],
        **summary,
        "sampled_rows_for_model": int(len(X_model)),
    }
    with open(OUTPUT_DIR / "resumo_base.json", "w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, indent=2, ensure_ascii=False)

    return AnalysisArtifacts(
        metrics=metrics,
        class_distribution=summary["class_distribution"],
        top_feature_importances=feature_importances.head(15).round(6).to_dict(orient="records"),
        grouped_feature_importances=grouped_importances.head(10).round(6).to_dict(orient="records"),
        top_risk_factors=top_risk_factors,
        sampled_rows=int(len(X_model)),
        total_rows=summary["total_rows"],
        dropped_rows=summary["dropped_rows"],
    )


def print_summary(artifacts: AnalysisArtifacts) -> None:
    print("Analise concluida com sucesso.")
    print(f"Registros consolidados: {artifacts.total_rows:,}".replace(",", "."))
    print(f"Registros usados no modelo: {artifacts.sampled_rows:,}".replace(",", "."))
    print(f"Registros descartados: {artifacts.dropped_rows:,}".replace(",", "."))
    print(f"Acuracia: {artifacts.metrics['accuracy']:.4f}")
    print("Top 10 variaveis por importancia agregada:")
    for item in artifacts.grouped_feature_importances[:10]:
        print(f" - {item['variavel']}: {item['importancia']:.4f}")
    print("Top 10 atributos por importancia:")
    for item in artifacts.top_feature_importances[:10]:
        print(f" - {item['atributo']}: {item['importancia']:.4f}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    analysis_artifacts = run_analysis()
    print_summary(analysis_artifacts)
