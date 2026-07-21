from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.parquet"
)

DEFAULT_SOURCE_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.csv"
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


def parse_arguments() -> argparse.Namespace:
    """Lit les options de la ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Construit la couche Bronze à partir des "
            "mesures industrielles simulées."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["overwrite", "error_if_exists"],
        default="overwrite",
        help=(
            "overwrite remplace la couche Bronze existante ; "
            "error_if_exists bloque si elle existe déjà."
        ),
    )

    return parser.parse_args()


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


def load_source_data() -> tuple[pd.DataFrame, Path, str]:
    """Charge les données sources, de préférence en Parquet."""

    if DEFAULT_SOURCE_PARQUET.exists():
        source_path = DEFAULT_SOURCE_PARQUET
        source_format = "parquet"
        dataframe = pd.read_parquet(source_path)
    elif DEFAULT_SOURCE_CSV.exists():
        source_path = DEFAULT_SOURCE_CSV
        source_format = "csv"
        dataframe = pd.read_csv(source_path)
    else:
        raise FileNotFoundError(
            "Aucune donnée avec anomalies n'a été trouvée. "
            "Exécute d'abord simulation/inject_anomalies.py."
        )

    if dataframe.empty:
        raise ValueError(
            "Le fichier source ne contient aucune ligne."
        )

    required_columns = {
        "measurement_id",
        "timestamp",
        "equipment_id",
        "operating_state",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires manquantes : "
            f"{sorted(missing_columns)}"
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )

    if "ingestion_timestamp" in dataframe.columns:
        dataframe["ingestion_timestamp"] = pd.to_datetime(
            dataframe["ingestion_timestamp"],
            errors="coerce",
        )

    if dataframe["measurement_id"].isna().any():
        raise ValueError(
            "measurement_id contient des valeurs manquantes."
        )

    if dataframe["measurement_id"].duplicated().any():
        raise ValueError(
            "measurement_id contient des doublons."
        )

    return dataframe, source_path, source_format


def prepare_bronze_dataframe(
    source: pd.DataFrame,
    source_path: Path,
    source_format: str,
    source_sha256: str,
) -> tuple[pd.DataFrame, str, datetime]:
    """Ajoute les métadonnées techniques de la couche Bronze."""

    ingested_at_utc = datetime.now(timezone.utc)

    batch_id = (
        "BRONZE_"
        f"{source_sha256[:16].upper()}"
    )

    bronze = source.copy(deep=True)

    bronze["bronze_record_id"] = (
        bronze["measurement_id"].astype(str)
    )

    bronze["bronze_ingestion_batch_id"] = (
        batch_id
    )

    bronze["bronze_ingested_at_utc"] = pd.Timestamp(
        ingested_at_utc
    )

    bronze["bronze_source_file"] = (
        source_path.name
    )

    bronze["bronze_source_format"] = (
        source_format
    )

    bronze["bronze_source_sha256"] = (
        source_sha256
    )

    bronze["bronze_data_layer"] = "bronze"

    bronze["bronze_event_date"] = (
        bronze["timestamp"].dt.date
    )

    bronze["bronze_year"] = (
        bronze["timestamp"].dt.year.astype("int16")
    )

    bronze["bronze_month"] = (
        bronze["timestamp"].dt.month.astype("int8")
    )

    bronze["bronze_day"] = (
        bronze["timestamp"].dt.day.astype("int8")
    )

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

    original_columns = [
        column
        for column in bronze.columns
        if column not in metadata_columns
    ]

    bronze = bronze[
        original_columns + metadata_columns
    ]

    return bronze, batch_id, ingested_at_utc


def prepare_output_directory(
    mode: str,
) -> None:
    """Prépare le dossier de sortie Bronze."""

    if BRONZE_DATASET_DIRECTORY.exists():
        if mode == "error_if_exists":
            raise FileExistsError(
                "La couche Bronze existe déjà : "
                f"{BRONZE_DATASET_DIRECTORY}"
            )

        shutil.rmtree(
            BRONZE_DATASET_DIRECTORY
        )

    BRONZE_DATASET_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    BRONZE_MANIFEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_partitioned_dataset(
    bronze: pd.DataFrame,
) -> None:
    """Écrit un dataset Parquet partitionné au format Hive."""

    table = pa.Table.from_pandas(
        bronze,
        preserve_index=False,
    )

    ds.write_dataset(
        data=table,
        base_dir=str(
            BRONZE_DATASET_DIRECTORY
        ),
        format="parquet",
        partitioning=[
            "bronze_year",
            "bronze_month",
            "equipment_id",
        ],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=50_000,
        max_rows_per_group=50_000,
    )


def build_manifest(
    source: pd.DataFrame,
    bronze: pd.DataFrame,
    source_path: Path,
    source_format: str,
    source_sha256: str,
    batch_id: str,
    ingested_at_utc: datetime,
) -> dict[str, Any]:
    """Construit le manifeste d'ingestion Bronze."""

    partitions = (
        bronze[
            [
                "bronze_year",
                "bronze_month",
                "equipment_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "bronze_year",
                "bronze_month",
                "equipment_id",
            ]
        )
    )

    parquet_files = list(
        BRONZE_DATASET_DIRECTORY.rglob(
            "*.parquet"
        )
    )

    manifest = {
        "batch_id": batch_id,
        "data_layer": "bronze",
        "status": "SUCCESS",
        "ingested_at_utc": (
            ingested_at_utc.isoformat()
        ),
        "source": {
            "path": str(source_path),
            "file_name": source_path.name,
            "format": source_format,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "target": {
            "dataset_directory": str(
                BRONZE_DATASET_DIRECTORY
            ),
            "format": "parquet",
            "partitioning": [
                "bronze_year",
                "bronze_month",
                "equipment_id",
            ],
            "parquet_file_count": len(
                parquet_files
            ),
            "partition_count": len(
                partitions
            ),
        },
        "statistics": {
            "source_row_count": int(
                len(source)
            ),
            "bronze_row_count": int(
                len(bronze)
            ),
            "column_count": int(
                len(bronze.columns)
            ),
            "equipment_count": int(
                bronze["equipment_id"].nunique()
            ),
            "minimum_timestamp": (
                bronze["timestamp"]
                .min()
                .isoformat()
            ),
            "maximum_timestamp": (
                bronze["timestamp"]
                .max()
                .isoformat()
            ),
        },
        "quality_checks": {
            "measurement_id_missing": int(
                bronze[
                    "measurement_id"
                ].isna().sum()
            ),
            "measurement_id_duplicates": int(
                bronze[
                    "measurement_id"
                ].duplicated().sum()
            ),
            "timestamp_missing": int(
                bronze["timestamp"].isna().sum()
            ),
        },
    }

    return manifest


def write_manifest(
    manifest: dict[str, Any],
    batch_id: str,
) -> Path:
    """Enregistre le manifeste JSON."""

    manifest_path = (
        BRONZE_MANIFEST_DIRECTORY
        / f"{batch_id}.json"
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
    """Construit la couche Bronze."""

    arguments = parse_arguments()

    print(
        "Chargement des données sources..."
    )

    source, source_path, source_format = (
        load_source_data()
    )

    source_sha256 = calculate_sha256(
        source_path
    )

    bronze, batch_id, ingested_at_utc = (
        prepare_bronze_dataframe(
            source=source,
            source_path=source_path,
            source_format=source_format,
            source_sha256=source_sha256,
        )
    )

    print(
        "Préparation du dossier Bronze..."
    )

    prepare_output_directory(
        mode=arguments.mode
    )

    print(
        "Écriture du dataset Parquet partitionné..."
    )

    write_partitioned_dataset(
        bronze
    )

    manifest = build_manifest(
        source=source,
        bronze=bronze,
        source_path=source_path,
        source_format=source_format,
        source_sha256=source_sha256,
        batch_id=batch_id,
        ingested_at_utc=ingested_at_utc,
    )

    manifest_path = write_manifest(
        manifest=manifest,
        batch_id=batch_id,
    )

    print()
    print(
        "Construction de la couche Bronze réussie."
    )
    print(
        f"Batch : {batch_id}"
    )
    print(
        f"Lignes ingérées : {len(bronze):,}"
    )
    print(
        f"Partitions : "
        f"{manifest['target']['partition_count']}"
    )
    print(
        f"Fichiers Parquet : "
        f"{manifest['target']['parquet_file_count']}"
    )
    print(
        f"Dataset : "
        f"{BRONZE_DATASET_DIRECTORY}"
    )
    print(
        f"Manifeste : {manifest_path}"
    )


if __name__ == "__main__":
    main()