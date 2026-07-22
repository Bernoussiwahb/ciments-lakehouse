from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "industrial_measurements_clean"
)

HOURLY_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "equipment_hourly_kpis"
)

DAILY_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "equipment_daily_kpis"
)

EQUIPMENT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "equipment_period_summary.parquet"
)

MANIFEST_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "_manifests"
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
    / "gold_validation"
)


def read_dataset(directory: Path) -> pd.DataFrame:
    """Charge un dataset Parquet partitionné."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    if not list(directory.rglob("*.parquet")):
        raise FileNotFoundError(
            f"Aucun Parquet dans : {directory}"
        )

    dataset = ds.dataset(
        str(directory),
        format="parquet",
        partitioning="hive",
    )

    return dataset.to_table().to_pandas()


def normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise les colonnes temporelles."""

    result = dataframe.copy()

    for column in [
        "timestamp",
        "period_start",
        "minimum_timestamp",
        "maximum_timestamp",
        "gold_processed_at_utc",
    ]:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="raise",
                utc=(column == "gold_processed_at_utc"),
            )

    return result


def load_latest_manifest() -> tuple[dict, Path]:
    """Charge le manifeste Gold le plus récent."""

    if not MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Gold n'existe pas."
        )

    paths = sorted(
        MANIFEST_DIRECTORY.glob("GOLD_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not paths:
        raise FileNotFoundError(
            "Aucun manifeste Gold trouvé."
        )

    path = paths[0]
    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    return manifest, path


def validate_counts(
    silver: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict,
) -> None:
    """Valide les nombres de lignes attendus."""

    silver = silver.copy()
    silver["hour_start"] = (
        silver["timestamp"].dt.floor("h")
    )
    silver["day_start"] = (
        silver["timestamp"].dt.floor("D")
    )

    expected_hourly = (
        silver[
            ["equipment_id", "hour_start"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_daily = (
        silver[
            ["equipment_id", "day_start"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_equipment = int(
        silver["equipment_id"].nunique()
    )

    if len(hourly) != expected_hourly:
        raise ValueError(
            "Nombre de lignes horaires incorrect : "
            f"attendu={expected_hourly:,}, "
            f"trouvé={len(hourly):,}."
        )

    if len(daily) != expected_daily:
        raise ValueError(
            "Nombre de lignes quotidiennes incorrect : "
            f"attendu={expected_daily:,}, "
            f"trouvé={len(daily):,}."
        )

    if len(summary) != expected_equipment:
        raise ValueError(
            "Nombre de lignes de synthèse incorrect."
        )

    if int(
        manifest["targets"]["hourly"]["row_count"]
    ) != len(hourly):
        raise ValueError(
            "Nombre horaire incorrect dans le manifeste."
        )

    if int(
        manifest["targets"]["daily"]["row_count"]
    ) != len(daily):
        raise ValueError(
            "Nombre quotidien incorrect dans le manifeste."
        )


def validate_keys(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Valide les clés uniques."""

    if hourly.duplicated(
        ["equipment_id", "period_start"]
    ).any():
        raise ValueError(
            "Doublons dans les KPI horaires."
        )

    if daily.duplicated(
        ["equipment_id", "period_start"]
    ).any():
        raise ValueError(
            "Doublons dans les KPI quotidiens."
        )

    if summary["equipment_id"].duplicated().any():
        raise ValueError(
            "Doublons dans la synthèse équipement."
        )


def validate_totals(
    silver: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Valide la conservation des totaux principaux."""

    columns = [
        "production_tonnes",
        "electrical_energy_kwh",
        "thermal_energy_mj",
    ]

    for column in columns:
        expected = float(
            silver[column].sum()
        )

        tolerance = max(
            0.1,
            abs(expected) * 1e-7,
        )

        for table_name, dataframe in [
            ("horaire", hourly),
            ("quotidienne", daily),
            ("synthèse équipement", summary),
        ]:
            actual = float(
                dataframe[column].sum()
            )

            if not np.isclose(
                actual,
                expected,
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"Total {column} incohérent "
                    f"dans la table {table_name}."
                )


def validate_derived_metrics(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    """Valide les KPI calculés."""

    expected_specific = np.divide(
        dataframe["electrical_energy_kwh"],
        dataframe["production_tonnes"],
        out=np.full(len(dataframe), np.nan),
        where=(
            dataframe["production_tonnes"].to_numpy() > 0
        ),
    )

    if not np.allclose(
        dataframe["specific_energy_kwh_t"],
        expected_specific,
        rtol=0,
        atol=0.011,
        equal_nan=True,
    ):
        raise ValueError(
            "Consommation spécifique incohérente "
            f"dans la table {table_name}."
        )

    expected_anomaly_rate = (
        dataframe["anomaly_row_count"]
        / dataframe["measurement_count"]
        * 100
    )

    if not np.allclose(
        dataframe["anomaly_rate_pct"],
        expected_anomaly_rate,
        rtol=0,
        atol=0.011,
        equal_nan=True,
    ):
        raise ValueError(
            "Taux d'anomalie incohérent "
            f"dans la table {table_name}."
        )

    expected_has_anomaly = (
        dataframe["anomaly_row_count"] > 0
    )

    if not (
        dataframe["has_anomaly"].astype(bool)
        == expected_has_anomaly
    ).all():
        raise ValueError(
            "has_anomaly est incohérent "
            f"dans la table {table_name}."
        )

    invalid_running_rate = (
        (dataframe["running_rate_pct"] < 0)
        | (dataframe["running_rate_pct"] > 100.01)
    )

    if invalid_running_rate.any():
        raise ValueError(
            "running_rate_pct hors limites "
            f"dans la table {table_name}."
        )


def validate_anomalies(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    manifest: dict,
) -> int:
    """Valide le nombre d'anomalies agrégées."""

    if ANOMALY_TRUTH_PATH.exists():
        expected = len(
            pd.read_parquet(ANOMALY_TRUTH_PATH)
        )
    else:
        expected = 0

    hourly_count = int(
        hourly["anomaly_row_count"].sum()
    )

    daily_count = int(
        daily["anomaly_row_count"].sum()
    )

    manifest_count = int(
        manifest["statistics"]["total_anomaly_rows"]
    )

    if hourly_count != expected:
        raise ValueError(
            "Nombre d'anomalies incorrect dans les KPI horaires."
        )

    if daily_count != expected:
        raise ValueError(
            "Nombre d'anomalies incorrect dans les KPI quotidiens."
        )

    if manifest_count != expected:
        raise ValueError(
            "Nombre d'anomalies incorrect dans le manifeste."
        )

    return expected


def validate_metadata(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Valide les métadonnées Gold."""

    required = {
        "gold_processing_batch_id",
        "gold_processed_at_utc",
        "gold_source_silver_batch_id",
        "gold_data_layer",
    }

    for table_name, dataframe in [
        ("horaire", hourly),
        ("quotidienne", daily),
        ("synthèse", summary),
    ]:
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Métadonnées manquantes dans {table_name} : "
                f"{sorted(missing)}"
            )

        if dataframe[
            "gold_processing_batch_id"
        ].nunique() != 1:
            raise ValueError(
                f"Plusieurs batchs dans la table {table_name}."
            )

        if set(
            dataframe["gold_data_layer"]
        ) != {"gold"}:
            raise ValueError(
                f"gold_data_layer incorrect dans {table_name}."
            )


def write_report(
    silver: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    anomaly_count: int,
    manifest: dict,
    manifest_path: Path,
) -> Path:
    """Écrit le rapport de validation Gold."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily[
        [
            "equipment_id",
            "period_start",
            "production_tonnes",
            "electrical_energy_kwh",
            "specific_energy_kwh_t",
            "running_rate_pct",
            "anomaly_row_count",
        ]
    ].sort_values(
        ["period_start", "equipment_id"]
    ).to_csv(
        REPORT_DIRECTORY
        / "gold_daily_kpi_preview.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    summary.to_csv(
        REPORT_DIRECTORY
        / "gold_equipment_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    lines = [
        "RAPPORT DE VALIDATION DE LA COUCHE GOLD",
        "=" * 45,
        "Statut : SUCCÈS",
        f"Batch : {manifest['batch_id']}",
        f"Manifeste : {manifest_path}",
        f"Lignes Silver sources : {len(silver):,}",
        f"Lignes KPI horaires : {len(hourly):,}",
        f"Lignes KPI quotidiennes : {len(daily):,}",
        f"Équipements synthétisés : {len(summary)}",
        f"Lignes anormales agrégées : {anomaly_count:,}",
        (
            "Production totale : "
            f"{daily['production_tonnes'].sum():,.3f} tonnes"
        ),
        (
            "Énergie électrique totale : "
            f"{daily['electrical_energy_kwh'].sum():,.3f} kWh"
        ),
        "",
        "Contrôles réussis :",
        "- nombre de groupes horaires et quotidiens",
        "- unicité des clés équipement/période",
        "- conservation de la production totale",
        "- conservation de l'énergie totale",
        "- conservation des anomalies métier",
        "- validation des KPI dérivés",
        "- métadonnées Gold complètes",
    ]

    report_path = (
        REPORT_DIRECTORY
        / "gold_validation_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

    return report_path


def main() -> None:
    """Valide la couche Gold."""

    print("Chargement des couches Silver et Gold...")

    silver = normalize_dates(
        read_dataset(SILVER_DIRECTORY)
    )

    hourly = normalize_dates(
        read_dataset(HOURLY_DIRECTORY)
    )

    daily = normalize_dates(
        read_dataset(DAILY_DIRECTORY)
    )

    if not EQUIPMENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "La synthèse équipement Gold est introuvable."
        )

    summary = normalize_dates(
        pd.read_parquet(EQUIPMENT_SUMMARY_PATH)
    )

    manifest, manifest_path = (
        load_latest_manifest()
    )

    validate_counts(
        silver,
        hourly,
        daily,
        summary,
        manifest,
    )

    validate_keys(
        hourly,
        daily,
        summary,
    )

    validate_totals(
        silver,
        hourly,
        daily,
        summary,
    )

    validate_derived_metrics(
        hourly,
        "horaire",
    )

    validate_derived_metrics(
        daily,
        "quotidienne",
    )

    anomaly_count = validate_anomalies(
        hourly,
        daily,
        manifest,
    )

    validate_metadata(
        hourly,
        daily,
        summary,
    )

    report_path = write_report(
        silver,
        hourly,
        daily,
        summary,
        anomaly_count,
        manifest,
        manifest_path,
    )

    print()
    print("Validation de la couche Gold réussie.")
    print(f"Lignes KPI horaires : {len(hourly):,}")
    print(f"Lignes KPI quotidiennes : {len(daily):,}")
    print(f"Équipements synthétisés : {len(summary)}")
    print(f"Anomalies agrégées : {anomaly_count:,}")
    print(f"Rapport : {report_path}")


if __name__ == "__main__":
    main()
