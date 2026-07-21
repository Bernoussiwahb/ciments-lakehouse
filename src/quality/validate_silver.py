from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "industrial_measurements"
)

SILVER_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "industrial_measurements_clean"
)

SILVER_QUARANTINE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "quarantine"
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

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "silver_validation"
)


def read_partitioned_dataset(
    directory: Path,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Charge un dataset Parquet partitionné."""

    if not directory.exists():
        if allow_empty:
            return pd.DataFrame()

        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    parquet_files = list(
        directory.rglob("*.parquet")
    )

    if not parquet_files:
        if allow_empty:
            return pd.DataFrame()

        raise FileNotFoundError(
            f"Aucun Parquet dans : {directory}"
        )

    dataset = ds.dataset(
        str(directory),
        format="parquet",
        partitioning="hive",
    )

    return dataset.to_table().to_pandas()


def load_latest_manifest() -> tuple[dict, Path]:
    """Charge le dernier manifeste Silver."""

    if not SILVER_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Silver n'existe pas."
        )

    paths = sorted(
        SILVER_MANIFEST_DIRECTORY.glob(
            "SILVER_*.json"
        ),
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


def normalize_datetime_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise les colonnes de date pour les comparaisons."""

    result = dataframe.copy()

    for column in [
        "timestamp",
        "ingestion_timestamp",
        "bronze_ingested_at_utc",
        "silver_processed_at_utc",
    ]:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
                utc=(
                    column.endswith("_utc")
                ),
            )

    return result


def validate_counts(
    bronze: pd.DataFrame,
    silver: pd.DataFrame,
    quarantine: pd.DataFrame,
    manifest: dict,
) -> None:
    """Vérifie la conservation des lignes."""

    if (
        len(silver)
        + len(quarantine)
        != len(bronze)
    ):
        raise ValueError(
            "Silver + quarantaine ne correspond pas "
            "au nombre de lignes Bronze."
        )

    stats = manifest["statistics"]

    expected = {
        "input_row_count": len(bronze),
        "valid_row_count": len(silver),
        "rejected_row_count": len(quarantine),
    }

    for key, value in expected.items():
        if int(stats[key]) != int(value):
            raise ValueError(
                f"Valeur incorrecte dans le manifeste : {key}."
            )


def validate_silver_structure(
    silver: pd.DataFrame,
) -> None:
    """Valide les clés, statuts et métadonnées Silver."""

    required_columns = {
        "measurement_id",
        "timestamp",
        "equipment_id",
        "silver_quality_issues",
        "silver_quality_issue_count",
        "silver_is_valid",
        "silver_quality_status",
        "silver_processing_batch_id",
        "silver_processed_at_utc",
        "silver_source_bronze_batch_id",
        "silver_data_layer",
        "silver_event_date",
        "silver_year",
        "silver_month",
        "silver_day",
        "silver_hour",
        "silver_day_of_week",
    }

    missing = (
        required_columns
        - set(silver.columns)
    )

    if missing:
        raise ValueError(
            "Colonnes Silver manquantes : "
            f"{sorted(missing)}"
        )

    if silver["measurement_id"].isna().any():
        raise ValueError(
            "measurement_id contient des valeurs nulles."
        )

    if silver["measurement_id"].duplicated().any():
        raise ValueError(
            "measurement_id contient des doublons."
        )

    if not silver["silver_is_valid"].all():
        raise ValueError(
            "Le dataset Silver propre contient "
            "des lignes invalides."
        )

    if set(
        silver["silver_quality_status"]
    ) != {"PASS"}:
        raise ValueError(
            "Le statut du dataset propre doit être PASS."
        )

    if set(
        silver["silver_data_layer"]
    ) != {"silver"}:
        raise ValueError(
            "silver_data_layer doit être égal à silver."
        )

    if silver[
        "silver_processing_batch_id"
    ].nunique() != 1:
        raise ValueError(
            "Plusieurs batchs sont présents dans Silver."
        )


def validate_formulas(
    silver: pd.DataFrame,
) -> None:
    """Valide les relations mathématiques principales."""

    expected_production = (
        silver["production_rate_tph"]
        * (5 / 60)
    )

    if not np.allclose(
        silver["production_tonnes"],
        expected_production,
        rtol=0,
        atol=0.011,
        equal_nan=True,
    ):
        raise ValueError(
            "La formule débit/production est incorrecte."
        )

    expected_energy = (
        silver["electrical_power_kw"]
        * (5 / 60)
    )

    if not np.allclose(
        silver["electrical_energy_kwh"],
        expected_energy,
        rtol=0,
        atol=0.011,
        equal_nan=True,
    ):
        raise ValueError(
            "La formule puissance/énergie est incorrecte."
        )

    positive_production = (
        silver["production_tonnes"] > 0
    )

    expected_specific = (
        silver.loc[
            positive_production,
            "electrical_energy_kwh",
        ]
        / silver.loc[
            positive_production,
            "production_tonnes",
        ]
    )

    if not np.allclose(
        silver.loc[
            positive_production,
            "specific_energy_kwh_t",
        ],
        expected_specific,
        rtol=0,
        atol=0.011,
        equal_nan=True,
    ):
        raise ValueError(
            "La consommation spécifique est incohérente."
        )


def validate_source_fidelity(
    bronze: pd.DataFrame,
    silver: pd.DataFrame,
) -> None:
    """
    Vérifie que Silver normalise sans altérer
    les mesures métier valides.
    """

    bronze_indexed = (
        bronze
        .set_index("measurement_id")
    )

    silver_indexed = (
        silver
        .set_index("measurement_id")
    )

    common_ids = silver_indexed.index

    columns_to_compare = [
        "timestamp",
        "equipment_id",
        "operating_state",
        "production_rate_tph",
        "production_tonnes",
        "electrical_power_kw",
        "electrical_energy_kwh",
        "specific_energy_kwh_t",
        "temperature_c",
        "pressure_bar",
        "vibration_mm_s",
        "alarm_count",
    ]

    for column in columns_to_compare:
        before = bronze_indexed.loc[
            common_ids,
            column,
        ]

        after = silver_indexed.loc[
            common_ids,
            column,
        ]

        if pd.api.types.is_numeric_dtype(before):
            equal = np.allclose(
                before.to_numpy(dtype=float),
                after.to_numpy(dtype=float),
                rtol=0,
                atol=0.0001,
                equal_nan=True,
            )
        elif pd.api.types.is_datetime64_any_dtype(before):
            equal = np.array_equal(
                pd.to_datetime(before).astype("int64"),
                pd.to_datetime(after).astype("int64"),
            )
        else:
            equal = (
                before.fillna("<NULL>")
                .astype(str)
                .equals(
                    after.fillna("<NULL>")
                    .astype(str)
                )
            )

        if not equal:
            raise ValueError(
                "Une mesure métier valide a été altérée "
                f"dans Silver : {column}"
            )


def validate_business_anomalies_preserved(
    silver: pd.DataFrame,
) -> tuple[int, int]:
    """Vérifie que les anomalies métier n'ont pas été rejetées."""

    if not ANOMALY_TRUTH_PATH.exists():
        return 0, 0

    truth = pd.read_parquet(
        ANOMALY_TRUTH_PATH
    )

    truth_ids = set(
        truth["measurement_id"]
    )

    silver_ids = set(
        silver["measurement_id"]
    )

    preserved_ids = (
        truth_ids & silver_ids
    )

    if preserved_ids != truth_ids:
        missing_count = len(
            truth_ids - silver_ids
        )

        raise ValueError(
            "Certaines anomalies métier ont été rejetées "
            f"par Silver : {missing_count} lignes."
        )

    return (
        len(truth_ids),
        len(preserved_ids),
    )


def write_report(
    bronze: pd.DataFrame,
    silver: pd.DataFrame,
    quarantine: pd.DataFrame,
    manifest: dict,
    manifest_path: Path,
    anomaly_count: int,
    preserved_anomaly_count: int,
) -> Path:
    """Écrit le rapport de validation Silver."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_summary = (
        silver.groupby(
            [
                "silver_year",
                "silver_month",
                "equipment_id",
            ],
            as_index=False,
        )
        .agg(
            row_count=(
                "measurement_id",
                "size",
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
            [
                "silver_year",
                "silver_month",
                "equipment_id",
            ]
        )
    )

    partition_summary.to_csv(
        REPORT_DIRECTORY
        / "silver_partition_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    lines = [
        "RAPPORT DE VALIDATION DE LA COUCHE SILVER",
        "=" * 45,
        "Statut : SUCCÈS",
        f"Batch : {manifest['batch_id']}",
        f"Manifeste : {manifest_path}",
        f"Lignes Bronze : {len(bronze):,}",
        f"Lignes Silver valides : {len(silver):,}",
        f"Lignes en quarantaine : {len(quarantine):,}",
        (
            "Taux de validité : "
            f"{len(silver) / len(bronze):.2%}"
        ),
        (
            "Nombre de partitions : "
            f"{len(partition_summary)}"
        ),
        (
            "Doublons measurement_id : "
            f"{silver['measurement_id'].duplicated().sum()}"
        ),
        (
            "Anomalies métier attendues : "
            f"{anomaly_count:,}"
        ),
        (
            "Anomalies métier conservées : "
            f"{preserved_anomaly_count:,}"
        ),
        "",
        "Contrôles réussis :",
        "- normalisation des types",
        "- conservation du nombre total de lignes",
        "- séparation valide/quarantaine",
        "- conservation des mesures métier valides",
        "- validation des formules calculées",
        "- conservation des anomalies métier",
        "- métadonnées Silver complètes",
    ]

    report_path = (
        REPORT_DIRECTORY
        / "silver_validation_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

    return report_path


def main() -> None:
    """Valide entièrement la couche Silver."""

    print(
        "Chargement des couches Bronze et Silver..."
    )

    bronze = normalize_datetime_columns(
        read_partitioned_dataset(
            BRONZE_DATASET_DIRECTORY
        )
    )

    silver = normalize_datetime_columns(
        read_partitioned_dataset(
            SILVER_DATASET_DIRECTORY
        )
    )

    quarantine = normalize_datetime_columns(
        read_partitioned_dataset(
            SILVER_QUARANTINE_DIRECTORY,
            allow_empty=True,
        )
    )

    manifest, manifest_path = (
        load_latest_manifest()
    )

    validate_counts(
        bronze=bronze,
        silver=silver,
        quarantine=quarantine,
        manifest=manifest,
    )

    validate_silver_structure(
        silver
    )

    validate_formulas(
        silver
    )

    validate_source_fidelity(
        bronze=bronze,
        silver=silver,
    )

    (
        anomaly_count,
        preserved_anomaly_count,
    ) = validate_business_anomalies_preserved(
        silver
    )

    report_path = write_report(
        bronze=bronze,
        silver=silver,
        quarantine=quarantine,
        manifest=manifest,
        manifest_path=manifest_path,
        anomaly_count=anomaly_count,
        preserved_anomaly_count=preserved_anomaly_count,
    )

    print()
    print(
        "Validation de la couche Silver réussie."
    )
    print(
        f"Lignes Silver : {len(silver):,}"
    )
    print(
        f"Lignes en quarantaine : "
        f"{len(quarantine):,}"
    )
    print(
        f"Anomalies métier conservées : "
        f"{preserved_anomaly_count:,}/"
        f"{anomaly_count:,}"
    )
    print(
        f"Rapport : {report_path}"
    )


if __name__ == "__main__":
    main()