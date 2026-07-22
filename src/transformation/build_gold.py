from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIRECTORY = (
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

GOLD_ROOT = PROJECT_ROOT / "data" / "gold"

HOURLY_DIRECTORY = (
    GOLD_ROOT / "equipment_hourly_kpis"
)

DAILY_DIRECTORY = (
    GOLD_ROOT / "equipment_daily_kpis"
)

EQUIPMENT_SUMMARY_PATH = (
    GOLD_ROOT / "equipment_period_summary.parquet"
)

MANIFEST_DIRECTORY = GOLD_ROOT / "_manifests"

INTERVAL_MINUTES = 5


def read_partitioned_dataset(directory: Path) -> pd.DataFrame:
    """Charge un dataset Parquet partitionné."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    if not list(directory.rglob("*.parquet")):
        raise FileNotFoundError(
            f"Aucun fichier Parquet trouvé dans : {directory}"
        )

    dataset = ds.dataset(
        str(directory),
        format="parquet",
        partitioning="hive",
    )

    dataframe = dataset.to_table().to_pandas()

    if dataframe.empty:
        raise ValueError(
            f"Le dataset est vide : {directory}"
        )

    return dataframe


def load_latest_silver_manifest() -> tuple[dict, Path]:
    """Charge le manifeste Silver le plus récent."""

    if not SILVER_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Silver n'existe pas."
        )

    paths = sorted(
        SILVER_MANIFEST_DIRECTORY.glob("SILVER_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not paths:
        raise FileNotFoundError(
            "Aucun manifeste Silver trouvé."
        )

    path = paths[0]
    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    return manifest, path


def load_silver() -> pd.DataFrame:
    """Charge et contrôle la couche Silver."""

    silver = read_partitioned_dataset(
        SILVER_DIRECTORY
    )

    silver["timestamp"] = pd.to_datetime(
        silver["timestamp"],
        errors="raise",
    )

    if silver["measurement_id"].duplicated().any():
        raise ValueError(
            "La couche Silver contient des identifiants dupliqués."
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

    return silver


def add_anomaly_truth(silver: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les étiquettes d'anomalie aux mesures Silver."""

    enriched = silver.copy(deep=True)

    if not ANOMALY_TRUTH_PATH.exists():
        enriched["is_anomaly"] = 0
        enriched["anomaly_id"] = pd.NA
        enriched["anomaly_type"] = pd.NA
        enriched["severity"] = pd.NA
        return enriched

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

    missing = required_columns - set(truth.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes dans la vérité terrain : "
            f"{sorted(missing)}"
        )

    if truth["measurement_id"].duplicated().any():
        raise ValueError(
            "La vérité terrain contient des identifiants dupliqués."
        )

    truth = truth[
        [
            "measurement_id",
            "is_anomaly",
            "anomaly_id",
            "anomaly_type",
            "severity",
        ]
    ]

    enriched = enriched.merge(
        truth,
        on="measurement_id",
        how="left",
        validate="one_to_one",
    )

    enriched["is_anomaly"] = (
        enriched["is_anomaly"]
        .fillna(0)
        .astype("int8")
    )

    return enriched


def prepare_helpers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les indicateurs nécessaires aux agrégations."""

    prepared = dataframe.copy(deep=True)

    state_mapping = {
        "RUNNING": "running_minutes_value",
        "MAINTENANCE": "maintenance_minutes_value",
        "STOPPED": "stopped_minutes_value",
        "IDLE": "idle_minutes_value",
    }

    for state, column in state_mapping.items():
        prepared[column] = np.where(
            prepared["operating_state"].eq(state),
            INTERVAL_MINUTES,
            0,
        )

    prepared["alarm_count"] = (
        pd.to_numeric(
            prepared["alarm_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    return prepared


def aggregate_period(
    dataframe: pd.DataFrame,
    frequency: str,
    granularity: str,
) -> pd.DataFrame:
    """Calcule les KPI par équipement et période."""

    prepared = dataframe.copy(deep=True)
    prepared["period_start"] = (
        prepared["timestamp"].dt.floor(frequency)
    )

    group_columns = [
        "equipment_id",
        "period_start",
    ]

    result = (
        prepared.groupby(
            group_columns,
            as_index=False,
            observed=True,
        )
        .agg(
            measurement_count=("measurement_id", "size"),
            running_minutes=("running_minutes_value", "sum"),
            maintenance_minutes=("maintenance_minutes_value", "sum"),
            stopped_minutes=("stopped_minutes_value", "sum"),
            idle_minutes=("idle_minutes_value", "sum"),
            production_tonnes=("production_tonnes", "sum"),
            average_production_rate_tph=("production_rate_tph", "mean"),
            maximum_production_rate_tph=("production_rate_tph", "max"),
            electrical_energy_kwh=("electrical_energy_kwh", "sum"),
            average_electrical_power_kw=("electrical_power_kw", "mean"),
            peak_electrical_power_kw=("electrical_power_kw", "max"),
            thermal_energy_mj=("thermal_energy_mj", "sum"),
            average_capacity_utilization_pct=("capacity_utilization_pct", "mean"),
            average_temperature_c=("temperature_c", "mean"),
            maximum_temperature_c=("temperature_c", "max"),
            average_pressure_bar=("pressure_bar", "mean"),
            minimum_pressure_bar=("pressure_bar", "min"),
            average_vibration_mm_s=("vibration_mm_s", "mean"),
            maximum_vibration_mm_s=("vibration_mm_s", "max"),
            average_current_ampere=("current_ampere", "mean"),
            average_rotation_speed_rpm=("rotation_speed_rpm", "mean"),
            alarm_count_total=("alarm_count", "sum"),
            anomaly_row_count=("is_anomaly", "sum"),
        )
    )

    anomaly_details = (
        prepared.groupby(
            group_columns,
            observed=True,
        )
        .agg(
            anomaly_type_count=(
                "anomaly_type",
                lambda values: int(
                    values.dropna().nunique()
                ),
            ),
            anomaly_event_count=(
                "anomaly_id",
                lambda values: int(
                    values.dropna().nunique()
                ),
            ),
            anomaly_types=(
                "anomaly_type",
                lambda values: ";".join(
                    sorted(
                        {
                            str(value)
                            for value in values.dropna()
                            if str(value) != "<NA>"
                        }
                    )
                ),
            ),
        )
        .reset_index()
    )

    result = result.merge(
        anomaly_details,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    total_minutes = (
        result["measurement_count"]
        * INTERVAL_MINUTES
    )

    result["running_rate_pct"] = (
        result["running_minutes"]
        / total_minutes
        * 100
    )

    result["specific_energy_kwh_t"] = np.divide(
        result["electrical_energy_kwh"],
        result["production_tonnes"],
        out=np.full(len(result), np.nan),
        where=(
            result["production_tonnes"].to_numpy() > 0
        ),
    )

    result["anomaly_rate_pct"] = (
        result["anomaly_row_count"]
        / result["measurement_count"]
        * 100
    )

    result["has_anomaly"] = (
        result["anomaly_row_count"] > 0
    )

    result["gold_granularity"] = granularity
    result["gold_event_date"] = (
        result["period_start"].dt.date
    )
    result["gold_year"] = (
        result["period_start"].dt.year.astype("int16")
    )
    result["gold_month"] = (
        result["period_start"].dt.month.astype("int8")
    )
    result["gold_day"] = (
        result["period_start"].dt.day.astype("int8")
    )

    if granularity == "hourly":
        result["gold_hour"] = (
            result["period_start"].dt.hour.astype("int8")
        )
    else:
        result["gold_hour"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int8",
        )

    numeric_columns = result.select_dtypes(
        include=["number"]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns].round(3)
    )

    return result.sort_values(
        ["equipment_id", "period_start"]
    ).reset_index(drop=True)


def aggregate_equipment_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Crée une synthèse sur toute la période par équipement."""

    summary = (
        dataframe.groupby(
            "equipment_id",
            as_index=False,
            observed=True,
        )
        .agg(
            measurement_count=("measurement_id", "size"),
            minimum_timestamp=("timestamp", "min"),
            maximum_timestamp=("timestamp", "max"),
            production_tonnes=("production_tonnes", "sum"),
            electrical_energy_kwh=("electrical_energy_kwh", "sum"),
            thermal_energy_mj=("thermal_energy_mj", "sum"),
            average_production_rate_tph=("production_rate_tph", "mean"),
            average_electrical_power_kw=("electrical_power_kw", "mean"),
            peak_electrical_power_kw=("electrical_power_kw", "max"),
            average_temperature_c=("temperature_c", "mean"),
            maximum_temperature_c=("temperature_c", "max"),
            average_vibration_mm_s=("vibration_mm_s", "mean"),
            maximum_vibration_mm_s=("vibration_mm_s", "max"),
            alarm_count_total=("alarm_count", "sum"),
            anomaly_row_count=("is_anomaly", "sum"),
            anomaly_event_count=(
                "anomaly_id",
                lambda values: int(
                    values.dropna().nunique()
                ),
            ),
        )
    )

    summary["specific_energy_kwh_t"] = np.divide(
        summary["electrical_energy_kwh"],
        summary["production_tonnes"],
        out=np.full(len(summary), np.nan),
        where=(
            summary["production_tonnes"].to_numpy() > 0
        ),
    )

    summary["anomaly_rate_pct"] = (
        summary["anomaly_row_count"]
        / summary["measurement_count"]
        * 100
    )

    numeric_columns = summary.select_dtypes(
        include=["number"]
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns].round(3)
    )

    return summary.sort_values(
        "equipment_id"
    ).reset_index(drop=True)


def add_gold_metadata(
    dataframe: pd.DataFrame,
    gold_batch_id: str,
    processed_at_utc: datetime,
    silver_batch_id: str,
) -> pd.DataFrame:
    """Ajoute les métadonnées communes Gold."""

    result = dataframe.copy(deep=True)
    result["gold_processing_batch_id"] = gold_batch_id
    result["gold_processed_at_utc"] = pd.Timestamp(
        processed_at_utc
    )
    result["gold_source_silver_batch_id"] = (
        silver_batch_id
    )
    result["gold_data_layer"] = "gold"

    return result


def prepare_outputs() -> None:
    """Réinitialise les sorties Gold générées."""

    for directory in [
        HOURLY_DIRECTORY,
        DAILY_DIRECTORY,
    ]:
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if EQUIPMENT_SUMMARY_PATH.exists():
        EQUIPMENT_SUMMARY_PATH.unlink()

    MANIFEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_dataset(
    dataframe: pd.DataFrame,
    directory: Path,
) -> int:
    """Écrit un dataset Parquet partitionné."""

    table = pa.Table.from_pandas(
        dataframe,
        preserve_index=False,
    )

    ds.write_dataset(
        data=table,
        base_dir=str(directory),
        format="parquet",
        partitioning=[
            "gold_year",
            "gold_month",
            "equipment_id",
        ],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=50_000,
        max_rows_per_group=50_000,
    )

    return len(
        list(directory.rglob("*.parquet"))
    )


def write_manifest(
    silver: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    silver_manifest: dict,
    silver_manifest_path: Path,
    gold_batch_id: str,
    processed_at_utc: datetime,
    hourly_file_count: int,
    daily_file_count: int,
) -> Path:
    """Écrit le manifeste Gold."""

    manifest = {
        "batch_id": gold_batch_id,
        "data_layer": "gold",
        "status": "SUCCESS",
        "processed_at_utc": processed_at_utc.isoformat(),
        "source": {
            "silver_batch_id": silver_manifest["batch_id"],
            "silver_manifest_path": str(
                silver_manifest_path
            ),
            "silver_dataset_directory": str(
                SILVER_DIRECTORY
            ),
        },
        "targets": {
            "hourly": {
                "directory": str(HOURLY_DIRECTORY),
                "row_count": int(len(hourly)),
                "file_count": hourly_file_count,
            },
            "daily": {
                "directory": str(DAILY_DIRECTORY),
                "row_count": int(len(daily)),
                "file_count": daily_file_count,
            },
            "equipment_summary": {
                "path": str(EQUIPMENT_SUMMARY_PATH),
                "row_count": int(len(summary)),
            },
        },
        "statistics": {
            "silver_measurement_count": int(len(silver)),
            "equipment_count": int(
                silver["equipment_id"].nunique()
            ),
            "minimum_timestamp": (
                silver["timestamp"].min().isoformat()
            ),
            "maximum_timestamp": (
                silver["timestamp"].max().isoformat()
            ),
            "total_production_tonnes": round(
                float(silver["production_tonnes"].sum()),
                3,
            ),
            "total_electrical_energy_kwh": round(
                float(
                    silver["electrical_energy_kwh"].sum()
                ),
                3,
            ),
            "total_anomaly_rows": int(
                silver["is_anomaly"].sum()
            ),
        },
    }

    path = (
        MANIFEST_DIRECTORY
        / f"{gold_batch_id}.json"
    )

    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    """Construit la couche Gold."""

    print("Chargement de la couche Silver...")

    silver_manifest, silver_manifest_path = (
        load_latest_silver_manifest()
    )

    silver = load_silver()

    print("Ajout de la vérité terrain des anomalies...")

    enriched = add_anomaly_truth(silver)
    prepared = prepare_helpers(enriched)

    processed_at_utc = datetime.now(timezone.utc)

    batch_seed = (
        str(silver_manifest["batch_id"])
        + processed_at_utc.isoformat()
    )

    gold_batch_id = (
        "GOLD_"
        + hashlib.sha256(
            batch_seed.encode("utf-8")
        ).hexdigest()[:16].upper()
    )

    print("Calcul des KPI horaires...")
    hourly = aggregate_period(
        prepared,
        frequency="h",
        granularity="hourly",
    )

    print("Calcul des KPI quotidiens...")
    daily = aggregate_period(
        prepared,
        frequency="D",
        granularity="daily",
    )

    print("Calcul de la synthèse par équipement...")
    summary = aggregate_equipment_summary(prepared)

    hourly = add_gold_metadata(
        hourly,
        gold_batch_id,
        processed_at_utc,
        str(silver_manifest["batch_id"]),
    )

    daily = add_gold_metadata(
        daily,
        gold_batch_id,
        processed_at_utc,
        str(silver_manifest["batch_id"]),
    )

    summary = add_gold_metadata(
        summary,
        gold_batch_id,
        processed_at_utc,
        str(silver_manifest["batch_id"]),
    )

    prepare_outputs()

    print("Écriture des tables Gold...")

    hourly_file_count = write_dataset(
        hourly,
        HOURLY_DIRECTORY,
    )

    daily_file_count = write_dataset(
        daily,
        DAILY_DIRECTORY,
    )

    summary.to_parquet(
        EQUIPMENT_SUMMARY_PATH,
        index=False,
    )

    manifest_path = write_manifest(
        silver=prepared,
        hourly=hourly,
        daily=daily,
        summary=summary,
        silver_manifest=silver_manifest,
        silver_manifest_path=silver_manifest_path,
        gold_batch_id=gold_batch_id,
        processed_at_utc=processed_at_utc,
        hourly_file_count=hourly_file_count,
        daily_file_count=daily_file_count,
    )

    print()
    print("Construction de la couche Gold réussie.")
    print(f"Batch : {gold_batch_id}")
    print(f"Lignes Silver traitées : {len(prepared):,}")
    print(f"Lignes KPI horaires : {len(hourly):,}")
    print(f"Lignes KPI quotidiennes : {len(daily):,}")
    print(f"Équipements synthétisés : {len(summary)}")
    print(
        "Anomalies agrégées : "
        f"{int(prepared['is_anomaly'].sum()):,}"
    )
    print(f"Manifeste : {manifest_path}")


if __name__ == "__main__":
    main()
