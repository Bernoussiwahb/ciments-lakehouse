from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "industrial_measurements_clean"
)

SILVER_MANIFEST_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "_manifests"
)

ANOMALY_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "anomaly_truth.parquet"
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

TRAIN_END_TIMESTAMP = pd.Timestamp(
    "2026-04-30 23:55:00"
)

ROLLING_WINDOW_1H = 12
ROLLING_WINDOW_6H = 72

BASE_MODEL_FEATURES = [
    "production_rate_tph",
    "capacity_utilization_pct",
    "electrical_power_kw",
    "electrical_energy_kwh",
    "specific_energy_kwh_t",
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "current_ampere",
    "rotation_speed_rpm",
    "alarm_count",
]

ENGINEERED_MODEL_FEATURES = [
    "power_per_production_rate",
    "temperature_delta_5m",
    "pressure_delta_5m",
    "vibration_delta_5m",
    "power_delta_5m",
    "production_delta_5m",
    "temperature_rolling_mean_1h",
    "temperature_rolling_std_1h",
    "pressure_rolling_mean_1h",
    "pressure_rolling_std_1h",
    "vibration_rolling_mean_1h",
    "vibration_rolling_std_1h",
    "power_rolling_mean_1h",
    "power_rolling_std_1h",
    "production_rolling_mean_1h",
    "production_rolling_std_1h",
    "specific_energy_rolling_mean_1h",
    "specific_energy_rolling_std_1h",
    "temperature_rolling_mean_6h",
    "temperature_rolling_std_6h",
    "vibration_rolling_mean_6h",
    "vibration_rolling_std_6h",
    "power_rolling_mean_6h",
    "power_rolling_std_6h",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

MODEL_FEATURE_COLUMNS = (
    BASE_MODEL_FEATURES
    + ENGINEERED_MODEL_FEATURES
)


def read_partitioned_dataset(
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
            f"Aucun fichier Parquet trouvé dans : {directory}"
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

    if dataframe.empty:
        raise ValueError(
            "La couche Silver est vide."
        )

    return dataframe


def load_latest_silver_manifest() -> tuple[dict[str, Any], Path]:
    """Charge le dernier manifeste Silver."""

    if not SILVER_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Silver n'existe pas."
        )

    manifest_paths = sorted(
        SILVER_MANIFEST_DIRECTORY.glob("SILVER_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not manifest_paths:
        raise FileNotFoundError(
            "Aucun manifeste Silver trouvé."
        )

    manifest_path = manifest_paths[0]

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    return manifest, manifest_path


def load_silver_data() -> pd.DataFrame:
    """Charge et contrôle les mesures Silver."""

    silver = read_partitioned_dataset(
        SILVER_DATASET_DIRECTORY
    )

    silver["timestamp"] = pd.to_datetime(
        silver["timestamp"],
        errors="raise",
    )

    if silver["measurement_id"].duplicated().any():
        raise ValueError(
            "measurement_id contient des doublons dans Silver."
        )

    if "silver_is_valid" in silver.columns:
        invalid_count = int(
            (~silver["silver_is_valid"].astype(bool)).sum()
        )

        if invalid_count:
            raise ValueError(
                "Le dataset Silver propre contient "
                f"{invalid_count} ligne(s) invalide(s)."
            )

    return (
        silver
        .sort_values(
            [
                "equipment_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def add_ground_truth(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute les étiquettes uniquement pour l'évaluation.

    Ces colonnes ne font jamais partie des variables
    données au modèle Isolation Forest.
    """

    result = dataframe.copy(deep=True)

    if ANOMALY_TRUTH_PATH.exists():
        truth = pd.read_parquet(
            ANOMALY_TRUTH_PATH
        )

        required_columns = {
            "measurement_id",
            "is_anomaly",
            "anomaly_id",
            "anomaly_type",
            "severity",
        }

        missing_columns = (
            required_columns
            - set(truth.columns)
        )

        if missing_columns:
            raise ValueError(
                "Colonnes manquantes dans la vérité terrain : "
                f"{sorted(missing_columns)}"
            )

        if truth["measurement_id"].duplicated().any():
            raise ValueError(
                "La vérité terrain contient des doublons."
            )

        result = result.merge(
            truth[
                [
                    "measurement_id",
                    "is_anomaly",
                    "anomaly_id",
                    "anomaly_type",
                    "severity",
                ]
            ],
            on="measurement_id",
            how="left",
            validate="one_to_one",
        )
    else:
        result["is_anomaly"] = 0
        result["anomaly_id"] = pd.NA
        result["anomaly_type"] = pd.NA
        result["severity"] = pd.NA

    result["is_anomaly"] = (
        result["is_anomaly"]
        .fillna(0)
        .astype("int8")
    )

    for column in [
        "anomaly_id",
        "anomaly_type",
        "severity",
    ]:
        result[column] = (
            result[column]
            .astype("string")
        )

    return result


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> np.ndarray:
    """Calcule un ratio en évitant les divisions par zéro."""

    return np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.full(
            len(numerator),
            np.nan,
            dtype=float,
        ),
        where=(
            denominator.to_numpy(dtype=float) > 0
        ),
    )


def add_time_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute des variables temporelles cycliques."""

    result = dataframe.copy(deep=True)

    hour_fraction = (
        result["timestamp"].dt.hour
        + result["timestamp"].dt.minute / 60
    )

    result["hour_sin"] = np.sin(
        2 * np.pi * hour_fraction / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * hour_fraction / 24
    )

    day_of_week = (
        result["timestamp"].dt.dayofweek
    )

    result["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    result["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    return result


def add_equipment_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les deltas et statistiques glissantes."""

    result = dataframe.copy(deep=True)

    result["power_per_production_rate"] = safe_ratio(
        result["electrical_power_kw"],
        result["production_rate_tph"],
    )

    grouped = result.groupby(
        "equipment_id",
        sort=False,
        observed=True,
    )

    delta_mapping = {
        "temperature_c": "temperature_delta_5m",
        "pressure_bar": "pressure_delta_5m",
        "vibration_mm_s": "vibration_delta_5m",
        "electrical_power_kw": "power_delta_5m",
        "production_rate_tph": "production_delta_5m",
    }

    for source_column, target_column in delta_mapping.items():
        result[target_column] = (
            grouped[source_column]
            .diff()
        )

    rolling_mapping_1h = {
        "temperature_c": "temperature",
        "pressure_bar": "pressure",
        "vibration_mm_s": "vibration",
        "electrical_power_kw": "power",
        "production_rate_tph": "production",
        "specific_energy_kwh_t": "specific_energy",
    }

    for source_column, prefix in rolling_mapping_1h.items():
        result[
            f"{prefix}_rolling_mean_1h"
        ] = (
            grouped[source_column]
            .transform(
                lambda values: values.rolling(
                    window=ROLLING_WINDOW_1H,
                    min_periods=1,
                ).mean()
            )
        )

        result[
            f"{prefix}_rolling_std_1h"
        ] = (
            grouped[source_column]
            .transform(
                lambda values: values.rolling(
                    window=ROLLING_WINDOW_1H,
                    min_periods=2,
                ).std(ddof=0)
            )
        )

    rolling_mapping_6h = {
        "temperature_c": "temperature",
        "vibration_mm_s": "vibration",
        "electrical_power_kw": "power",
    }

    for source_column, prefix in rolling_mapping_6h.items():
        result[
            f"{prefix}_rolling_mean_6h"
        ] = (
            grouped[source_column]
            .transform(
                lambda values: values.rolling(
                    window=ROLLING_WINDOW_6H,
                    min_periods=1,
                ).mean()
            )
        )

        result[
            f"{prefix}_rolling_std_6h"
        ] = (
            grouped[source_column]
            .transform(
                lambda values: values.rolling(
                    window=ROLLING_WINDOW_6H,
                    min_periods=2,
                ).std(ddof=0)
            )
        )

    return result


def add_split_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Crée une séparation temporelle train/test."""

    result = dataframe.copy(deep=True)

    result["dataset_split"] = np.where(
        result["timestamp"] <= TRAIN_END_TIMESTAMP,
        "train",
        "test",
    )

    result["is_training_candidate"] = (
        result["dataset_split"].eq("train")
        & result["operating_state"].eq("RUNNING")
        & result["is_anomaly"].eq(0)
    )

    return result


def finalize_feature_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Nettoie uniquement les valeurs techniques des features."""

    result = dataframe.copy(deep=True)

    for feature in MODEL_FEATURE_COLUMNS:
        if feature not in result.columns:
            raise ValueError(
                f"Variable IA manquante : {feature}"
            )

        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce",
        )

        result[feature] = (
            result[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # Les premières lignes de chaque équipement n'ont pas
    # encore d'historique pour les deltas/écarts-types.
    # On utilise une valeur neutre 0 uniquement pour ces
    # variables dérivées techniques.
    for feature in ENGINEERED_MODEL_FEATURES:
        result[feature] = (
            result[feature]
            .fillna(0.0)
        )

    result["feature_event_date"] = (
        result["timestamp"].dt.date
    )

    result["feature_year"] = (
        result["timestamp"].dt.year.astype("int16")
    )

    result["feature_month"] = (
        result["timestamp"].dt.month.astype("int8")
    )

    result["feature_day"] = (
        result["timestamp"].dt.day.astype("int8")
    )

    numeric_columns = result.select_dtypes(
        include=["number"]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .round(6)
    )

    return result


def prepare_output_directory() -> None:
    """Réinitialise le dataset de variables IA."""

    if FEATURE_DATASET_DIRECTORY.exists():
        shutil.rmtree(
            FEATURE_DATASET_DIRECTORY
        )

    FEATURE_DATASET_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_MANIFEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_feature_dataset(
    dataframe: pd.DataFrame,
) -> int:
    """Écrit la table de variables en Parquet partitionné."""

    table = pa.Table.from_pandas(
        dataframe,
        preserve_index=False,
    )

    ds.write_dataset(
        data=table,
        base_dir=str(
            FEATURE_DATASET_DIRECTORY
        ),
        format="parquet",
        partitioning=[
            "feature_year",
            "feature_month",
            "equipment_id",
        ],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=50_000,
        max_rows_per_group=50_000,
    )

    return len(
        list(
            FEATURE_DATASET_DIRECTORY.rglob(
                "*.parquet"
            )
        )
    )


def write_feature_list() -> None:
    """Enregistre la liste officielle des variables du modèle."""

    payload = {
        "model_name": "IsolationForest",
        "model_scope": (
            "One model per equipment"
        ),
        "feature_count": len(
            MODEL_FEATURE_COLUMNS
        ),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "excluded_from_model": [
            "measurement_id",
            "timestamp",
            "equipment_id",
            "operating_state",
            "dataset_split",
            "is_training_candidate",
            "is_anomaly",
            "anomaly_id",
            "anomaly_type",
            "severity",
        ],
        "label_usage": (
            "Ground-truth columns are used only for "
            "controlled evaluation, never as model inputs."
        ),
    }

    FEATURE_LIST_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_manifest(
    feature_table: pd.DataFrame,
    silver_manifest: dict[str, Any],
    silver_manifest_path: Path,
    feature_batch_id: str,
    processed_at_utc: datetime,
    parquet_file_count: int,
) -> Path:
    """Écrit le manifeste de préparation des variables."""

    manifest = {
        "batch_id": feature_batch_id,
        "data_layer": "gold_ml_features",
        "status": "SUCCESS",
        "processed_at_utc": (
            processed_at_utc.isoformat()
        ),
        "source": {
            "silver_batch_id": (
                silver_manifest["batch_id"]
            ),
            "silver_manifest_path": str(
                silver_manifest_path
            ),
            "silver_dataset_directory": str(
                SILVER_DATASET_DIRECTORY
            ),
        },
        "target": {
            "dataset_directory": str(
                FEATURE_DATASET_DIRECTORY
            ),
            "feature_list_path": str(
                FEATURE_LIST_PATH
            ),
            "format": "parquet",
            "partitioning": [
                "feature_year",
                "feature_month",
                "equipment_id",
            ],
            "parquet_file_count": int(
                parquet_file_count
            ),
        },
        "statistics": {
            "row_count": int(
                len(feature_table)
            ),
            "equipment_count": int(
                feature_table[
                    "equipment_id"
                ].nunique()
            ),
            "model_feature_count": int(
                len(MODEL_FEATURE_COLUMNS)
            ),
            "train_row_count": int(
                feature_table[
                    "dataset_split"
                ].eq("train").sum()
            ),
            "test_row_count": int(
                feature_table[
                    "dataset_split"
                ].eq("test").sum()
            ),
            "training_candidate_count": int(
                feature_table[
                    "is_training_candidate"
                ].sum()
            ),
            "ground_truth_anomaly_count": int(
                feature_table[
                    "is_anomaly"
                ].sum()
            ),
            "minimum_timestamp": (
                feature_table[
                    "timestamp"
                ].min().isoformat()
            ),
            "maximum_timestamp": (
                feature_table[
                    "timestamp"
                ].max().isoformat()
            ),
        },
        "experimental_policy": {
            "train_end_timestamp": str(
                TRAIN_END_TIMESTAMP
            ),
            "training_candidate_rule": (
                "train split AND RUNNING state "
                "AND ground_truth_is_anomaly = 0"
            ),
            "warning": (
                "Ground truth is used to create a clean "
                "synthetic benchmark. This does not prove "
                "performance on real industrial data."
            ),
        },
    }

    manifest_path = (
        FEATURE_MANIFEST_DIRECTORY
        / f"{feature_batch_id}.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest_path


def main() -> None:
    """Construit la table de variables IA."""

    print(
        "Chargement de la couche Silver..."
    )

    silver_manifest, silver_manifest_path = (
        load_latest_silver_manifest()
    )

    silver = load_silver_data()

    print(
        "Ajout de la vérité terrain d'évaluation..."
    )

    feature_table = add_ground_truth(
        silver
    )

    print(
        "Création des variables temporelles..."
    )

    feature_table = add_time_features(
        feature_table
    )

    print(
        "Création des variables glissantes "
        "par équipement..."
    )

    feature_table = add_equipment_features(
        feature_table
    )

    feature_table = add_split_columns(
        feature_table
    )

    feature_table = finalize_feature_table(
        feature_table
    )

    processed_at_utc = datetime.now(
        timezone.utc
    )

    batch_seed = (
        str(silver_manifest["batch_id"])
        + processed_at_utc.isoformat()
    )

    feature_batch_id = (
        "FEATURES_"
        + hashlib.sha256(
            batch_seed.encode("utf-8")
        ).hexdigest()[:16].upper()
    )

    feature_table[
        "feature_processing_batch_id"
    ] = feature_batch_id

    feature_table[
        "feature_processed_at_utc"
    ] = pd.Timestamp(
        processed_at_utc
    )

    feature_table[
        "feature_source_silver_batch_id"
    ] = str(
        silver_manifest["batch_id"]
    )

    feature_table[
        "feature_data_layer"
    ] = "gold_ml_features"

    prepare_output_directory()

    print(
        "Écriture de la table de variables IA..."
    )

    parquet_file_count = (
        write_feature_dataset(
            feature_table
        )
    )

    write_feature_list()

    manifest_path = write_manifest(
        feature_table=feature_table,
        silver_manifest=silver_manifest,
        silver_manifest_path=silver_manifest_path,
        feature_batch_id=feature_batch_id,
        processed_at_utc=processed_at_utc,
        parquet_file_count=parquet_file_count,
    )

    print()
    print(
        "Construction de la table de variables IA réussie."
    )
    print(
        f"Lignes préparées : "
        f"{len(feature_table):,}"
    )
    print(
        f"Variables du modèle : "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )
    print(
        f"Lignes train : "
        f"{feature_table['dataset_split'].eq('train').sum():,}"
    )
    print(
        f"Lignes test : "
        f"{feature_table['dataset_split'].eq('test').sum():,}"
    )
    print(
        f"Candidats d'entraînement propres : "
        f"{feature_table['is_training_candidate'].sum():,}"
    )
    print(
        f"Anomalies conservées pour l'évaluation : "
        f"{feature_table['is_anomaly'].sum():,}"
    )
    print(
        f"Dataset : "
        f"{FEATURE_DATASET_DIRECTORY}"
    )
    print(
        f"Manifeste : {manifest_path}"
    )


if __name__ == "__main__":
    main()
