from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "industrial_measurements_clean"
)

FEATURE_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_feature_table"
)

FEATURE_MANIFEST_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "_feature_manifests"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_feature_columns.json"
)

ANOMALY_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "anomaly_truth.parquet"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "feature_validation"
)


def read_dataset(
    directory: Path,
) -> pd.DataFrame:
    """Charge un dataset Parquet partitionné."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    parquet_files = list(
        directory.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"Aucun Parquet trouvé dans : {directory}"
        )

    dataset = ds.dataset(
        str(directory),
        format="parquet",
        partitioning="hive",
    )

    dataframe = (
        dataset
        .to_table()
        .to_pandas()
    )

    return dataframe


def load_latest_manifest() -> tuple[dict, Path]:
    """Charge le manifeste de variables le plus récent."""

    if not FEATURE_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes de variables "
            "n'existe pas."
        )

    paths = sorted(
        FEATURE_MANIFEST_DIRECTORY.glob(
            "FEATURES_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not paths:
        raise FileNotFoundError(
            "Aucun manifeste de variables trouvé."
        )

    path = paths[0]

    manifest = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return manifest, path


def load_feature_list() -> list[str]:
    """Charge la liste officielle des variables du modèle."""

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            "ml_feature_columns.json est introuvable."
        )

    content = json.loads(
        FEATURE_LIST_PATH.read_text(
            encoding="utf-8"
        )
    )

    features = content.get(
        "feature_columns"
    )

    if not features:
        raise ValueError(
            "La liste des variables du modèle est vide."
        )

    return list(features)


def validate_counts(
    silver: pd.DataFrame,
    features: pd.DataFrame,
    manifest: dict,
) -> None:
    """Valide la conservation des lignes et des clés."""

    if len(features) != len(silver):
        raise ValueError(
            "Le nombre de lignes de la table IA "
            "diffère de Silver."
        )

    if features["measurement_id"].isna().any():
        raise ValueError(
            "measurement_id contient des valeurs nulles."
        )

    if features["measurement_id"].duplicated().any():
        raise ValueError(
            "measurement_id contient des doublons."
        )

    if set(
        features["measurement_id"]
    ) != set(
        silver["measurement_id"]
    ):
        raise ValueError(
            "Les identifiants Silver et Features "
            "ne correspondent pas."
        )

    if int(
        manifest["statistics"]["row_count"]
    ) != len(features):
        raise ValueError(
            "Le manifeste contient un nombre "
            "de lignes incorrect."
        )


def validate_model_features(
    features: pd.DataFrame,
    model_features: list[str],
) -> None:
    """Valide les colonnes réellement utilisables par le modèle."""

    missing_columns = (
        set(model_features)
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Variables IA manquantes : "
            f"{sorted(missing_columns)}"
        )

    feature_matrix = features[
        model_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    infinite_count = int(
        np.isinf(
            feature_matrix.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count:
        raise ValueError(
            f"La matrice contient {infinite_count} "
            "valeur(s) infinie(s)."
        )

    missing_by_feature = (
        feature_matrix
        .isna()
        .sum()
    )

    unexpected_missing = (
        missing_by_feature[
            missing_by_feature > 0
        ]
    )

    # specific_energy_kwh_t peut rester nulle lorsque
    # la production est nulle pendant maintenance.
    allowed_missing_features = {
        "specific_energy_kwh_t",
    }

    invalid_missing_features = set(
        unexpected_missing.index
    ) - allowed_missing_features

    if invalid_missing_features:
        raise ValueError(
            "Valeurs manquantes inattendues dans : "
            f"{sorted(invalid_missing_features)}"
        )

    training_candidates = features[
        features["is_training_candidate"].astype(bool)
    ]

    if training_candidates.empty:
        raise ValueError(
            "Aucune ligne candidate pour l'entraînement."
        )

    training_matrix = training_candidates[
        model_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if training_matrix.isna().any().any():
        missing_columns = (
            training_matrix.columns[
                training_matrix.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Les candidats d'entraînement contiennent "
            "des valeurs manquantes dans : "
            f"{missing_columns}"
        )


def validate_split(
    features: pd.DataFrame,
    manifest: dict,
) -> None:
    """Valide la séparation temporelle train/test."""

    valid_splits = {
        "train",
        "test",
    }

    actual_splits = set(
        features["dataset_split"]
    )

    if actual_splits != valid_splits:
        raise ValueError(
            "Valeurs dataset_split incorrectes : "
            f"{actual_splits}"
        )

    train = features[
        features["dataset_split"].eq("train")
    ]

    test = features[
        features["dataset_split"].eq("test")
    ]

    if train.empty or test.empty:
        raise ValueError(
            "Le train ou le test est vide."
        )

    if train["timestamp"].max() >= test["timestamp"].min():
        raise ValueError(
            "La séparation temporelle train/test "
            "n'est pas strictement ordonnée."
        )

    stats = manifest["statistics"]

    if int(
        stats["train_row_count"]
    ) != len(train):
        raise ValueError(
            "Nombre de lignes train incorrect "
            "dans le manifeste."
        )

    if int(
        stats["test_row_count"]
    ) != len(test):
        raise ValueError(
            "Nombre de lignes test incorrect "
            "dans le manifeste."
        )

    candidates = features[
        features["is_training_candidate"].astype(bool)
    ]

    invalid_candidates = candidates[
        ~candidates["dataset_split"].eq("train")
        | ~candidates["operating_state"].eq("RUNNING")
        | ~candidates["is_anomaly"].eq(0)
    ]

    if not invalid_candidates.empty:
        raise ValueError(
            "Certaines lignes candidates ne respectent "
            "pas la règle d'entraînement."
        )


def validate_labels(
    features: pd.DataFrame,
    manifest: dict,
) -> int:
    """Valide les étiquettes réservées à l'évaluation."""

    expected_anomalies = 0

    if ANOMALY_TRUTH_PATH.exists():
        truth = pd.read_parquet(
            ANOMALY_TRUTH_PATH
        )

        expected_anomalies = len(truth)

        truth_ids = set(
            truth["measurement_id"]
        )

        feature_anomaly_ids = set(
            features.loc[
                features["is_anomaly"].eq(1),
                "measurement_id",
            ]
        )

        if truth_ids != feature_anomaly_ids:
            raise ValueError(
                "Les anomalies de la table IA ne "
                "correspondent pas à la vérité terrain."
            )

    actual_anomalies = int(
        features["is_anomaly"].sum()
    )

    if actual_anomalies != expected_anomalies:
        raise ValueError(
            "Nombre d'anomalies incorrect "
            "dans la table IA."
        )

    if int(
        manifest["statistics"][
            "ground_truth_anomaly_count"
        ]
    ) != expected_anomalies:
        raise ValueError(
            "Nombre d'anomalies incorrect "
            "dans le manifeste."
        )

    return expected_anomalies


def validate_metadata(
    features: pd.DataFrame,
) -> None:
    """Valide les métadonnées de préparation."""

    required_columns = {
        "feature_processing_batch_id",
        "feature_processed_at_utc",
        "feature_source_silver_batch_id",
        "feature_data_layer",
        "feature_event_date",
        "feature_year",
        "feature_month",
        "feature_day",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Métadonnées de variables manquantes : "
            f"{sorted(missing_columns)}"
        )

    if features[
        "feature_processing_batch_id"
    ].nunique() != 1:
        raise ValueError(
            "Plusieurs batchs sont présents."
        )

    if set(
        features["feature_data_layer"]
    ) != {"gold_ml_features"}:
        raise ValueError(
            "feature_data_layer est incorrect."
        )


def write_report(
    features: pd.DataFrame,
    model_features: list[str],
    anomaly_count: int,
    manifest: dict,
    manifest_path: Path,
) -> Path:
    """Écrit le rapport de validation des variables."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    equipment_summary = (
        features.groupby(
            "equipment_id",
            as_index=False,
        )
        .agg(
            row_count=(
                "measurement_id",
                "size",
            ),
            training_candidate_count=(
                "is_training_candidate",
                "sum",
            ),
            anomaly_count=(
                "is_anomaly",
                "sum",
            ),
            minimum_timestamp=(
                "timestamp",
                "min",
            ),
            maximum_timestamp=(
                "timestamp",
                "max",
            ),
        )
        .sort_values(
            "equipment_id"
        )
    )

    equipment_summary.to_csv(
        REPORT_DIRECTORY
        / "feature_equipment_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    lines = [
        "RAPPORT DE VALIDATION DE LA TABLE DE VARIABLES IA",
        "=" * 52,
        "Statut : SUCCÈS",
        f"Batch : {manifest['batch_id']}",
        f"Manifeste : {manifest_path}",
        f"Lignes totales : {len(features):,}",
        (
            "Nombre d'équipements : "
            f"{features['equipment_id'].nunique()}"
        ),
        (
            "Variables du modèle : "
            f"{len(model_features)}"
        ),
        (
            "Lignes train : "
            f"{features['dataset_split'].eq('train').sum():,}"
        ),
        (
            "Lignes test : "
            f"{features['dataset_split'].eq('test').sum():,}"
        ),
        (
            "Candidats d'entraînement : "
            f"{features['is_training_candidate'].sum():,}"
        ),
        (
            "Anomalies d'évaluation : "
            f"{anomaly_count:,}"
        ),
        "",
        "Contrôles réussis :",
        "- conservation des measurement_id",
        "- conservation du nombre de lignes Silver",
        "- absence de valeurs infinies",
        "- matrice d'entraînement sans valeur manquante",
        "- séparation temporelle train/test",
        "- exclusion des anomalies du jeu d'entraînement",
        "- conservation exacte de la vérité terrain",
        "- métadonnées de préparation complètes",
        "",
        "Important :",
        (
            "Les colonnes is_anomaly, anomaly_type, "
            "anomaly_id et severity servent uniquement "
            "à l'évaluation et ne sont pas fournies au modèle."
        ),
    ]

    report_path = (
        REPORT_DIRECTORY
        / "feature_validation_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

    return report_path


def main() -> None:
    """Valide entièrement la table de variables IA."""

    print(
        "Chargement des données Silver "
        "et de la table de variables IA..."
    )

    silver = read_dataset(
        SILVER_DATASET_DIRECTORY
    )

    features = read_dataset(
        FEATURE_DATASET_DIRECTORY
    )

    for dataframe in [
        silver,
        features,
    ]:
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="raise",
        )

    manifest, manifest_path = (
        load_latest_manifest()
    )

    model_features = load_feature_list()

    validate_counts(
        silver=silver,
        features=features,
        manifest=manifest,
    )

    validate_model_features(
        features=features,
        model_features=model_features,
    )

    validate_split(
        features=features,
        manifest=manifest,
    )

    anomaly_count = validate_labels(
        features=features,
        manifest=manifest,
    )

    validate_metadata(
        features
    )

    report_path = write_report(
        features=features,
        model_features=model_features,
        anomaly_count=anomaly_count,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    print()
    print(
        "Validation de la table de variables IA réussie."
    )
    print(
        f"Lignes validées : {len(features):,}"
    )
    print(
        f"Variables du modèle : "
        f"{len(model_features)}"
    )
    print(
        f"Lignes train : "
        f"{features['dataset_split'].eq('train').sum():,}"
    )
    print(
        f"Lignes test : "
        f"{features['dataset_split'].eq('test').sum():,}"
    )
    print(
        f"Candidats d'entraînement : "
        f"{features['is_training_candidate'].sum():,}"
    )
    print(
        f"Anomalies conservées : "
        f"{anomaly_count:,}"
    )
    print(
        f"Rapport : {report_path}"
    )


if __name__ == "__main__":
    main()
