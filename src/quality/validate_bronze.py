from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.parquet"
)

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

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "bronze_validation"
)


def calculate_sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_source() -> pd.DataFrame:
    """Charge le fichier source Parquet."""

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source introuvable : {SOURCE_PATH}"
        )

    source = pd.read_parquet(
        SOURCE_PATH
    )

    source["timestamp"] = pd.to_datetime(
        source["timestamp"],
        errors="raise",
    )

    return source


def load_bronze() -> pd.DataFrame:
    """Charge le dataset Bronze partitionné."""

    if not BRONZE_DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "La couche Bronze n'existe pas. "
            "Exécute d'abord src/ingestion/build_bronze.py."
        )

    parquet_files = list(
        BRONZE_DATASET_DIRECTORY.rglob(
            "*.parquet"
        )
    )

    if not parquet_files:
        raise FileNotFoundError(
            "Aucun fichier Parquet trouvé "
            "dans la couche Bronze."
        )

    dataset = ds.dataset(
        str(BRONZE_DATASET_DIRECTORY),
        format="parquet",
        partitioning="hive",
    )

    bronze = dataset.to_table().to_pandas()

    bronze["timestamp"] = pd.to_datetime(
        bronze["timestamp"],
        errors="raise",
    )

    bronze["bronze_ingested_at_utc"] = (
        pd.to_datetime(
            bronze["bronze_ingested_at_utc"],
            errors="raise",
            utc=True,
        )
    )

    return bronze


def load_latest_manifest() -> tuple[dict, Path]:
    """Charge le manifeste Bronze le plus récent."""

    if not BRONZE_MANIFEST_DIRECTORY.exists():
        raise FileNotFoundError(
            "Le dossier des manifestes Bronze "
            "n'existe pas."
        )

    manifest_paths = sorted(
        BRONZE_MANIFEST_DIRECTORY.glob(
            "BRONZE_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not manifest_paths:
        raise FileNotFoundError(
            "Aucun manifeste Bronze trouvé."
        )

    manifest_path = manifest_paths[0]

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    return manifest, manifest_path


def validate_structure(
    source: pd.DataFrame,
    bronze: pd.DataFrame,
) -> None:
    """Valide la structure et les identifiants."""

    if len(source) != len(bronze):
        raise ValueError(
            "Le nombre de lignes diffère : "
            f"source={len(source):,}, "
            f"bronze={len(bronze):,}."
        )

    required_metadata_columns = {
        "bronze_record_id",
        "bronze_ingestion_batch_id",
        "bronze_ingested_at_utc",
        "bronze_source_file",
        "bronze_source_format",
        "bronze_source_sha256",
        "bronze_data_layer",
        "bronze_event_date",
        "bronze_year",
        "bronze_month",
        "bronze_day",
    }

    missing_columns = (
        required_metadata_columns
        - set(bronze.columns)
    )

    if missing_columns:
        raise ValueError(
            "Métadonnées Bronze manquantes : "
            f"{sorted(missing_columns)}"
        )

    if source["measurement_id"].duplicated().any():
        raise ValueError(
            "La source contient des identifiants dupliqués."
        )

    if bronze["measurement_id"].duplicated().any():
        raise ValueError(
            "La couche Bronze contient des identifiants "
            "dupliqués."
        )

    source_ids = set(
        source["measurement_id"]
    )

    bronze_ids = set(
        bronze["measurement_id"]
    )

    if source_ids != bronze_ids:
        raise ValueError(
            "Les measurement_id de la couche Bronze "
            "ne correspondent pas à la source."
        )

    if not (
        bronze["bronze_record_id"].astype(str)
        == bronze["measurement_id"].astype(str)
    ).all():
        raise ValueError(
            "bronze_record_id ne correspond pas "
            "à measurement_id."
        )


def validate_source_fidelity(
    source: pd.DataFrame,
    bronze: pd.DataFrame,
) -> None:
    """Vérifie que les colonnes sources n'ont pas été altérées."""

    bronze_aligned = (
        bronze
        .set_index("measurement_id")
        .loc[source["measurement_id"]]
        .reset_index()
    )

    source_aligned = source.reset_index(
        drop=True
    )

    ignored_columns = {
        "ingestion_timestamp",
    }

    columns_to_compare = [
        column
        for column in source.columns
        if column not in ignored_columns
    ]

    for column in columns_to_compare:
        left = source_aligned[column]
        right = bronze_aligned[column]

        if pd.api.types.is_datetime64_any_dtype(
            left
        ):
            left_values = pd.to_datetime(
                left
            ).astype("int64")

            right_values = pd.to_datetime(
                right
            ).astype("int64")

            equal = np.array_equal(
                left_values.to_numpy(),
                right_values.to_numpy(),
            )

        elif pd.api.types.is_numeric_dtype(
            left
        ):
            equal = np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0,
                atol=0.0001,
                equal_nan=True,
            )

        else:
            equal = (
                left.fillna("<NULL>")
                .astype(str)
                .equals(
                    right.fillna("<NULL>")
                    .astype(str)
                )
            )

        if not equal:
            raise ValueError(
                "La colonne source a été altérée "
                f"dans Bronze : {column}"
            )


def validate_metadata(
    source: pd.DataFrame,
    bronze: pd.DataFrame,
    manifest: dict,
) -> None:
    """Valide les métadonnées et le manifeste."""

    metadata_columns = [
        "bronze_record_id",
        "bronze_ingestion_batch_id",
        "bronze_ingested_at_utc",
        "bronze_source_file",
        "bronze_source_format",
        "bronze_source_sha256",
        "bronze_data_layer",
        "bronze_event_date",
        "bronze_year",
        "bronze_month",
        "bronze_day",
    ]

    for column in metadata_columns:
        if bronze[column].isna().any():
            raise ValueError(
                f"Valeurs manquantes dans {column}."
            )

    if bronze[
        "bronze_ingestion_batch_id"
    ].nunique() != 1:
        raise ValueError(
            "Plusieurs batch_id sont présents "
            "dans la couche Bronze."
        )

    if set(
        bronze["bronze_data_layer"]
    ) != {"bronze"}:
        raise ValueError(
            "bronze_data_layer doit être égal "
            "à 'bronze'."
        )

    source_hash = calculate_sha256(
        SOURCE_PATH
    )

    bronze_hashes = set(
        bronze["bronze_source_sha256"]
    )

    if bronze_hashes != {source_hash}:
        raise ValueError(
            "L'empreinte SHA-256 Bronze "
            "ne correspond pas à la source."
        )

    if (
        manifest["source"]["sha256"]
        != source_hash
    ):
        raise ValueError(
            "L'empreinte du manifeste "
            "ne correspond pas à la source."
        )

    if (
        manifest["statistics"]["source_row_count"]
        != len(source)
        or manifest["statistics"]["bronze_row_count"]
        != len(bronze)
    ):
        raise ValueError(
            "Le manifeste contient un nombre "
            "de lignes incorrect."
        )

    expected_year = source[
        "timestamp"
    ].dt.year.astype(int)

    expected_month = source[
        "timestamp"
    ].dt.month.astype(int)

    expected_day = source[
        "timestamp"
    ].dt.day.astype(int)

    bronze_aligned = (
        bronze
        .set_index("measurement_id")
        .loc[source["measurement_id"]]
        .reset_index()
    )

    if not np.array_equal(
        bronze_aligned[
            "bronze_year"
        ].astype(int).to_numpy(),
        expected_year.to_numpy(),
    ):
        raise ValueError(
            "bronze_year est incorrect."
        )

    if not np.array_equal(
        bronze_aligned[
            "bronze_month"
        ].astype(int).to_numpy(),
        expected_month.to_numpy(),
    ):
        raise ValueError(
            "bronze_month est incorrect."
        )

    if not np.array_equal(
        bronze_aligned[
            "bronze_day"
        ].astype(int).to_numpy(),
        expected_day.to_numpy(),
    ):
        raise ValueError(
            "bronze_day est incorrect."
        )


def build_partition_summary(
    bronze: pd.DataFrame,
) -> pd.DataFrame:
    """Construit un résumé des partitions Bronze."""

    summary = (
        bronze.groupby(
            [
                "bronze_year",
                "bronze_month",
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
                "bronze_year",
                "bronze_month",
                "equipment_id",
            ]
        )
    )

    return summary


def write_validation_report(
    source: pd.DataFrame,
    bronze: pd.DataFrame,
    manifest: dict,
    manifest_path: Path,
    partition_summary: pd.DataFrame,
) -> Path:
    """Écrit le rapport de validation Bronze."""

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_summary.to_csv(
        REPORT_DIRECTORY
        / "bronze_partition_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    lines = [
        "RAPPORT DE VALIDATION DE LA COUCHE BRONZE",
        "=" * 45,
        f"Statut : SUCCÈS",
        f"Batch : {manifest['batch_id']}",
        f"Manifeste : {manifest_path}",
        f"Lignes source : {len(source):,}",
        f"Lignes Bronze : {len(bronze):,}",
        (
            "Nombre d'équipements : "
            f"{bronze['equipment_id'].nunique()}"
        ),
        (
            "Nombre de partitions : "
            f"{len(partition_summary)}"
        ),
        (
            "Nombre de fichiers Parquet : "
            f"{len(list(BRONZE_DATASET_DIRECTORY.rglob('*.parquet')))}"
        ),
        (
            "Date minimale : "
            f"{bronze['timestamp'].min()}"
        ),
        (
            "Date maximale : "
            f"{bronze['timestamp'].max()}"
        ),
        (
            "Doublons measurement_id : "
            f"{bronze['measurement_id'].duplicated().sum()}"
        ),
        (
            "Valeurs manquantes measurement_id : "
            f"{bronze['measurement_id'].isna().sum()}"
        ),
        (
            "Empreinte source : "
            f"{manifest['source']['sha256']}"
        ),
        "",
        "Contrôles réussis :",
        "- conservation du nombre de lignes",
        "- conservation des measurement_id",
        "- conservation des valeurs sources",
        "- métadonnées Bronze complètes",
        "- partitionnement année/mois/équipement",
        "- manifeste cohérent avec la source",
    ]

    report_path = (
        REPORT_DIRECTORY
        / "bronze_validation_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

    return report_path


def main() -> None:
    """Valide entièrement la couche Bronze."""

    print(
        "Chargement de la source et de la couche Bronze..."
    )

    source = load_source()
    bronze = load_bronze()
    manifest, manifest_path = (
        load_latest_manifest()
    )

    validate_structure(
        source=source,
        bronze=bronze,
    )

    validate_source_fidelity(
        source=source,
        bronze=bronze,
    )

    validate_metadata(
        source=source,
        bronze=bronze,
        manifest=manifest,
    )

    partition_summary = (
        build_partition_summary(
            bronze
        )
    )

    report_path = write_validation_report(
        source=source,
        bronze=bronze,
        manifest=manifest,
        manifest_path=manifest_path,
        partition_summary=partition_summary,
    )

    print()
    print(
        "Validation de la couche Bronze réussie."
    )
    print(
        f"Lignes validées : {len(bronze):,}"
    )
    print(
        f"Partitions validées : "
        f"{len(partition_summary)}"
    )
    print(
        f"Rapport : {report_path}"
    )


if __name__ == "__main__":
    main()