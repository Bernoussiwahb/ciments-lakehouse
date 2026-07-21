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

BRONZE_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "industrial_measurements"
)

BRONZE_MANIFEST_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "_manifests"
)

SILVER_ROOT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
)

SILVER_DATASET_DIRECTORY = (
    SILVER_ROOT_DIRECTORY
    / "industrial_measurements_clean"
)

SILVER_QUARANTINE_DIRECTORY = (
    SILVER_ROOT_DIRECTORY
    / "quarantine"
)

SILVER_MANIFEST_DIRECTORY = (
    SILVER_ROOT_DIRECTORY
    / "_manifests"
)

ALLOWED_OPERATING_STATES = {
    "RUNNING",
    "MAINTENANCE",
    "STOPPED",
    "IDLE",
}

KEY_COLUMNS = [
    "measurement_id",
    "timestamp",
    "equipment_id",
    "site_id",
]

NON_NEGATIVE_COLUMNS = [
    "production_rate_tph",
    "production_tonnes",
    "capacity_utilization_pct",
    "electrical_power_kw",
    "electrical_energy_kwh",
    "thermal_energy_mj",
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "current_ampere",
    "rotation_speed_rpm",
    "alarm_count",
]


def load_latest_bronze_manifest() -> tuple[dict[str, Any], Path]:
    """Charge le manifeste Bronze le plus récent."""

    if not BRONZE_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Bronze n'existe pas."
        )

    manifest_paths = sorted(
        BRONZE_MANIFEST_DIRECTORY.glob("BRONZE_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not manifest_paths:
        raise FileNotFoundError(
            "Aucun manifeste Bronze n'a été trouvé."
        )

    manifest_path = manifest_paths[0]
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    return manifest, manifest_path


def load_bronze_data() -> pd.DataFrame:
    """Charge le dataset Bronze partitionné."""

    if not BRONZE_DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "La couche Bronze n'existe pas. "
            "Exécute d'abord src/ingestion/build_bronze.py."
        )

    parquet_files = list(
        BRONZE_DATASET_DIRECTORY.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            "Aucun fichier Parquet trouvé dans Bronze."
        )

    dataset = ds.dataset(
        str(BRONZE_DATASET_DIRECTORY),
        format="parquet",
        partitioning="hive",
    )

    bronze = dataset.to_table().to_pandas()

    if bronze.empty:
        raise ValueError(
            "La couche Bronze est vide."
        )

    return bronze


def normalize_types(
    bronze: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise les types sans corriger les valeurs métier."""

    silver = bronze.copy(deep=True)

    datetime_columns = [
        "timestamp",
        "ingestion_timestamp",
        "bronze_ingested_at_utc",
    ]

    for column in datetime_columns:
        if column in silver.columns:
            silver[column] = pd.to_datetime(
                silver[column],
                errors="coerce",
                utc=(
                    column
                    == "bronze_ingested_at_utc"
                ),
            )

    string_columns = [
        "measurement_id",
        "site_id",
        "workshop_id",
        "equipment_id",
        "shift",
        "operating_state",
        "bronze_record_id",
        "bronze_ingestion_batch_id",
        "bronze_source_file",
        "bronze_source_format",
        "bronze_source_sha256",
        "bronze_data_layer",
    ]

    for column in string_columns:
        if column in silver.columns:
            silver[column] = (
                silver[column]
                .astype("string")
                .str.strip()
            )

    numeric_columns = [
        column
        for column in NON_NEGATIVE_COLUMNS
        if column in silver.columns
    ]

    if "specific_energy_kwh_t" in silver.columns:
        numeric_columns.append(
            "specific_energy_kwh_t"
        )

    for column in numeric_columns:
        silver[column] = pd.to_numeric(
            silver[column],
            errors="coerce",
        )

    if "alarm_count" in silver.columns:
        silver["alarm_count"] = (
            silver["alarm_count"]
            .round()
            .astype("Int64")
        )

    return silver


def append_issue(
    issues: pd.Series,
    mask: pd.Series,
    issue_code: str,
) -> pd.Series:
    """Ajoute un code d'erreur aux lignes concernées."""

    mask = mask.fillna(False)

    current = issues.loc[mask]

    issues.loc[mask] = np.where(
        current.eq(""),
        issue_code,
        current + ";" + issue_code,
    )

    return issues


def build_quality_flags(
    silver: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique uniquement des contrôles techniques.

    Les anomalies métier réalistes ne sont pas rejetées :
    surchauffe, vibration élevée ou surconsommation restent
    des observations valides si elles sont correctement mesurées.
    """

    checked = silver.copy(deep=True)

    issues = pd.Series(
        "",
        index=checked.index,
        dtype="string",
    )

    for column in KEY_COLUMNS:
        if column not in checked.columns:
            issues = append_issue(
                issues,
                pd.Series(True, index=checked.index),
                f"MISSING_COLUMN_{column.upper()}",
            )
        else:
            issues = append_issue(
                issues,
                checked[column].isna()
                | checked[column].astype("string").str.len().eq(0),
                f"MISSING_{column.upper()}",
            )

    if "measurement_id" in checked.columns:
        duplicate_mask = checked[
            "measurement_id"
        ].duplicated(keep=False)

        issues = append_issue(
            issues,
            duplicate_mask,
            "DUPLICATE_MEASUREMENT_ID",
        )

    if "operating_state" in checked.columns:
        invalid_state_mask = (
            checked["operating_state"].isna()
            | ~checked["operating_state"].isin(
                ALLOWED_OPERATING_STATES
            )
        )

        issues = append_issue(
            issues,
            invalid_state_mask,
            "INVALID_OPERATING_STATE",
        )

    for column in NON_NEGATIVE_COLUMNS:
        if column in checked.columns:
            issues = append_issue(
                issues,
                checked[column].notna()
                & checked[column].lt(0),
                f"NEGATIVE_{column.upper()}",
            )

    numeric_measure_columns = [
        column
        for column in [
            *NON_NEGATIVE_COLUMNS,
            "specific_energy_kwh_t",
        ]
        if column in checked.columns
    ]

    for column in numeric_measure_columns:
        issues = append_issue(
            issues,
            checked[column].notna()
            & ~np.isfinite(
                checked[column].astype(float)
            ),
            f"NON_FINITE_{column.upper()}",
        )

    if {
        "operating_state",
        "production_tonnes",
        "specific_energy_kwh_t",
    }.issubset(checked.columns):
        expected_specific_null = (
            checked["production_tonnes"].fillna(0).le(0)
            | checked["operating_state"].eq("MAINTENANCE")
        )

        unexpected_specific_null = (
            checked["specific_energy_kwh_t"].isna()
            & ~expected_specific_null
        )

        issues = append_issue(
            issues,
            unexpected_specific_null,
            "UNEXPECTED_NULL_SPECIFIC_ENERGY",
        )

    if {
        "production_rate_tph",
        "production_tonnes",
    }.issubset(checked.columns):
        expected_production = (
            checked["production_rate_tph"]
            * (5 / 60)
        )

        production_formula_error = (
            checked["production_rate_tph"].notna()
            & checked["production_tonnes"].notna()
            & ~np.isclose(
                checked["production_tonnes"],
                expected_production,
                rtol=0,
                atol=0.011,
            )
        )

        issues = append_issue(
            issues,
            production_formula_error,
            "INCONSISTENT_PRODUCTION_FORMULA",
        )

    if {
        "electrical_power_kw",
        "electrical_energy_kwh",
    }.issubset(checked.columns):
        expected_energy = (
            checked["electrical_power_kw"]
            * (5 / 60)
        )

        energy_formula_error = (
            checked["electrical_power_kw"].notna()
            & checked["electrical_energy_kwh"].notna()
            & ~np.isclose(
                checked["electrical_energy_kwh"],
                expected_energy,
                rtol=0,
                atol=0.011,
            )
        )

        issues = append_issue(
            issues,
            energy_formula_error,
            "INCONSISTENT_ENERGY_FORMULA",
        )

    if {
        "production_tonnes",
        "electrical_energy_kwh",
        "specific_energy_kwh_t",
    }.issubset(checked.columns):
        positive_production = (
            checked["production_tonnes"].gt(0)
        )

        expected_specific = np.divide(
            checked["electrical_energy_kwh"],
            checked["production_tonnes"],
            out=np.full(
                len(checked),
                np.nan,
                dtype=float,
            ),
            where=positive_production.to_numpy(),
        )

        specific_formula_error = (
            positive_production
            & checked["specific_energy_kwh_t"].notna()
            & ~np.isclose(
                checked["specific_energy_kwh_t"],
                expected_specific,
                rtol=0,
                atol=0.011,
            )
        )

        issues = append_issue(
            issues,
            specific_formula_error,
            "INCONSISTENT_SPECIFIC_ENERGY_FORMULA",
        )

    checked["silver_quality_issues"] = issues

    checked["silver_quality_issue_count"] = (
        issues.fillna("")
        .str.count(";")
        .add(
            issues.fillna("").ne("").astype(int)
        )
        .astype("int16")
    )

    checked["silver_is_valid"] = (
        checked["silver_quality_issue_count"].eq(0)
    )

    checked["silver_quality_status"] = np.where(
        checked["silver_is_valid"],
        "PASS",
        "REJECT",
    )

    return checked


def add_silver_metadata(
    checked: pd.DataFrame,
    bronze_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, str, datetime]:
    """Ajoute les métadonnées de traitement Silver."""

    processed_at_utc = datetime.now(timezone.utc)

    bronze_batch_id = str(
        bronze_manifest["batch_id"]
    )

    silver_batch_seed = (
        bronze_batch_id
        + processed_at_utc.isoformat()
    )

    silver_batch_id = (
        "SILVER_"
        + hashlib.sha256(
            silver_batch_seed.encode("utf-8")
        ).hexdigest()[:16].upper()
    )

    result = checked.copy(deep=True)

    result["silver_processing_batch_id"] = (
        silver_batch_id
    )

    result["silver_processed_at_utc"] = (
        pd.Timestamp(processed_at_utc)
    )

    result["silver_source_bronze_batch_id"] = (
        bronze_batch_id
    )

    result["silver_data_layer"] = "silver"

    result["silver_event_date"] = (
        result["timestamp"].dt.date
    )

    result["silver_year"] = (
        result["timestamp"].dt.year.astype("Int16")
    )

    result["silver_month"] = (
        result["timestamp"].dt.month.astype("Int8")
    )

    result["silver_day"] = (
        result["timestamp"].dt.day.astype("Int8")
    )

    result["silver_hour"] = (
        result["timestamp"].dt.hour.astype("Int8")
    )

    result["silver_day_of_week"] = (
        result["timestamp"].dt.dayofweek.astype("Int8")
    )

    return (
        result,
        silver_batch_id,
        processed_at_utc,
    )


def prepare_output_directories() -> None:
    """Réinitialise les sorties Silver générées."""

    for directory in [
        SILVER_DATASET_DIRECTORY,
        SILVER_QUARANTINE_DIRECTORY,
    ]:
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    SILVER_MANIFEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_partitioned_dataset(
    dataframe: pd.DataFrame,
    directory: Path,
    partition_columns: list[str],
) -> int:
    """Écrit un dataset Parquet partitionné."""

    if dataframe.empty:
        return 0

    table = pa.Table.from_pandas(
        dataframe,
        preserve_index=False,
    )

    ds.write_dataset(
        data=table,
        base_dir=str(directory),
        format="parquet",
        partitioning=partition_columns,
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=50_000,
        max_rows_per_group=50_000,
    )

    return len(
        list(directory.rglob("*.parquet"))
    )


def build_quality_summary(
    checked: pd.DataFrame,
) -> pd.DataFrame:
    """Résume les erreurs techniques détectées."""

    issue_rows: list[dict[str, Any]] = []

    for issue_string in checked[
        "silver_quality_issues"
    ].fillna(""):
        if not issue_string:
            continue

        for issue_code in issue_string.split(";"):
            issue_rows.append(
                {"issue_code": issue_code}
            )

    if not issue_rows:
        return pd.DataFrame(
            columns=[
                "issue_code",
                "affected_row_count",
            ]
        )

    return (
        pd.DataFrame(issue_rows)
        .value_counts("issue_code")
        .rename("affected_row_count")
        .reset_index()
        .sort_values(
            [
                "affected_row_count",
                "issue_code",
            ],
            ascending=[False, True],
        )
    )


def write_manifest(
    checked: pd.DataFrame,
    valid: pd.DataFrame,
    rejected: pd.DataFrame,
    quality_summary: pd.DataFrame,
    bronze_manifest: dict[str, Any],
    bronze_manifest_path: Path,
    silver_batch_id: str,
    processed_at_utc: datetime,
    valid_file_count: int,
    rejected_file_count: int,
) -> Path:
    """Crée le manifeste du traitement Silver."""

    manifest = {
        "batch_id": silver_batch_id,
        "data_layer": "silver",
        "status": "SUCCESS",
        "processed_at_utc": (
            processed_at_utc.isoformat()
        ),
        "source": {
            "bronze_batch_id": (
                bronze_manifest["batch_id"]
            ),
            "bronze_manifest_path": str(
                bronze_manifest_path
            ),
            "bronze_dataset_directory": str(
                BRONZE_DATASET_DIRECTORY
            ),
        },
        "target": {
            "valid_dataset_directory": str(
                SILVER_DATASET_DIRECTORY
            ),
            "quarantine_directory": str(
                SILVER_QUARANTINE_DIRECTORY
            ),
            "format": "parquet",
            "partitioning": [
                "silver_year",
                "silver_month",
                "equipment_id",
            ],
            "valid_file_count": valid_file_count,
            "rejected_file_count": rejected_file_count,
        },
        "statistics": {
            "input_row_count": int(
                len(checked)
            ),
            "valid_row_count": int(
                len(valid)
            ),
            "rejected_row_count": int(
                len(rejected)
            ),
            "validity_rate_pct": round(
                len(valid)
                / len(checked)
                * 100,
                4,
            ),
            "equipment_count": int(
                valid["equipment_id"].nunique()
                if not valid.empty
                else 0
            ),
            "minimum_timestamp": (
                valid["timestamp"].min().isoformat()
                if not valid.empty
                else None
            ),
            "maximum_timestamp": (
                valid["timestamp"].max().isoformat()
                if not valid.empty
                else None
            ),
        },
        "quality_issues": (
            quality_summary.to_dict(
                orient="records"
            )
        ),
        "quality_policy": {
            "principle": (
                "Only technical data-quality errors are rejected. "
                "Business anomalies remain valid observations."
            ),
            "expected_specific_energy_null": (
                "Allowed when production is zero or state is MAINTENANCE."
            ),
        },
    }

    manifest_path = (
        SILVER_MANIFEST_DIRECTORY
        / f"{silver_batch_id}.json"
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
    """Construit la couche Silver."""

    print(
        "Chargement de la couche Bronze..."
    )

    bronze_manifest, bronze_manifest_path = (
        load_latest_bronze_manifest()
    )

    bronze = load_bronze_data()

    print(
        "Normalisation des types..."
    )

    normalized = normalize_types(
        bronze
    )

    print(
        "Application des contrôles qualité techniques..."
    )

    checked = build_quality_flags(
        normalized
    )

    (
        checked,
        silver_batch_id,
        processed_at_utc,
    ) = add_silver_metadata(
        checked,
        bronze_manifest,
    )

    valid = (
        checked[
            checked["silver_is_valid"]
        ]
        .copy()
        .sort_values(
            [
                "equipment_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    rejected = (
        checked[
            ~checked["silver_is_valid"]
        ]
        .copy()
        .sort_values(
            [
                "equipment_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    quality_summary = build_quality_summary(
        checked
    )

    prepare_output_directories()

    print(
        "Écriture du dataset Silver valide..."
    )

    valid_file_count = write_partitioned_dataset(
        dataframe=valid,
        directory=SILVER_DATASET_DIRECTORY,
        partition_columns=[
            "silver_year",
            "silver_month",
            "equipment_id",
        ],
    )

    print(
        "Écriture de la zone de quarantaine..."
    )

    rejected_file_count = (
        write_partitioned_dataset(
            dataframe=rejected,
            directory=SILVER_QUARANTINE_DIRECTORY,
            partition_columns=[
                "silver_year",
                "silver_month",
                "equipment_id",
            ],
        )
    )

    manifest_path = write_manifest(
        checked=checked,
        valid=valid,
        rejected=rejected,
        quality_summary=quality_summary,
        bronze_manifest=bronze_manifest,
        bronze_manifest_path=bronze_manifest_path,
        silver_batch_id=silver_batch_id,
        processed_at_utc=processed_at_utc,
        valid_file_count=valid_file_count,
        rejected_file_count=rejected_file_count,
    )

    print()
    print(
        "Construction de la couche Silver réussie."
    )
    print(
        f"Batch : {silver_batch_id}"
    )
    print(
        f"Lignes reçues : {len(checked):,}"
    )
    print(
        f"Lignes valides : {len(valid):,}"
    )
    print(
        f"Lignes rejetées : {len(rejected):,}"
    )
    print(
        f"Taux de validité : "
        f"{len(valid) / len(checked):.2%}"
    )
    print(
        f"Dataset Silver : "
        f"{SILVER_DATASET_DIRECTORY}"
    )
    print(
        f"Manifeste : {manifest_path}"
    )


if __name__ == "__main__":
    main()