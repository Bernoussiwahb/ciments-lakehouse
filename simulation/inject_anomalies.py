from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.yml"
EQUIPMENT_PATH = PROJECT_ROOT / "data" / "reference" / "equipment.csv"

NORMAL_PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_normal.parquet"
)

NORMAL_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_normal.csv"
)

ANOMALOUS_PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.parquet"
)

ANOMALOUS_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.csv"
)

LABELS_DIRECTORY = PROJECT_ROOT / "data" / "labels"

ANOMALY_LABELS_PATH = (
    LABELS_DIRECTORY / "anomaly_labels.csv"
)

ANOMALY_TRUTH_PATH = (
    LABELS_DIRECTORY / "anomaly_truth.parquet"
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Charge un fichier YAML."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not content:
        raise ValueError(
            f"Le fichier YAML est vide : {path}"
        )

    return content


def load_normal_data() -> pd.DataFrame:
    """Charge les données normales, de préférence en Parquet."""

    if NORMAL_PARQUET_PATH.exists():
        dataframe = pd.read_parquet(
            NORMAL_PARQUET_PATH
        )
    elif NORMAL_CSV_PATH.exists():
        dataframe = pd.read_csv(
            NORMAL_CSV_PATH,
            parse_dates=[
                "timestamp",
                "ingestion_timestamp",
            ],
        )
    else:
        raise FileNotFoundError(
            "Aucun fichier de mesures normales trouvé. "
            "Exécute d'abord generate_normal_data.py."
        )

    if dataframe.empty:
        raise ValueError(
            "Le jeu de données normales est vide."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )

    dataframe["ingestion_timestamp"] = pd.to_datetime(
        dataframe["ingestion_timestamp"],
        errors="raise",
    )

    if dataframe["measurement_id"].duplicated().any():
        raise ValueError(
            "Les données normales contiennent des "
            "measurement_id dupliqués."
        )

    # On conserve l'ordre original du fichier normal.
    # Les recherches de fenêtres trient localement les données
    # sans modifier l'ordre global du jeu de données.
    return dataframe.reset_index(drop=True)


def load_equipment_reference() -> pd.DataFrame:
    """Charge la référence des équipements."""

    if not EQUIPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {EQUIPMENT_PATH}"
        )

    equipment = pd.read_csv(
        EQUIPMENT_PATH
    )

    required_columns = {
        "equipment_id",
        "nominal_capacity_tph",
    }

    missing_columns = (
        required_columns
        - set(equipment.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans equipment.csv : "
            f"{sorted(missing_columns)}"
        )

    return equipment


def get_interval(
    config: dict[str, Any],
) -> pd.Timedelta:
    """Retourne la fréquence temporelle de la simulation."""

    frequency = config["simulation"]["frequency"]
    interval = pd.Timedelta(frequency)

    if interval <= pd.Timedelta(0):
        raise ValueError(
            "La fréquence doit être positive."
        )

    return interval


def build_candidate_offsets(
    maximum_offset: int,
) -> list[int]:
    """Construit l'ordre 0, +1, -1, +2, -2, etc."""

    offsets = [0]

    for value in range(1, maximum_offset + 1):
        offsets.extend([value, -value])

    return offsets


def find_running_window(
    dataframe: pd.DataFrame,
    equipment_id: str,
    target_fraction: float,
    duration: pd.Timedelta,
    interval: pd.Timedelta,
    occupied_measurement_ids: set[str],
) -> pd.Index:
    """
    Trouve une fenêtre continue en état RUNNING.

    La recherche commence près d'une position cible dans
    la période totale, puis se déplace si nécessaire.
    """

    equipment_data = dataframe[
        dataframe["equipment_id"] == equipment_id
    ].sort_values("timestamp")

    if equipment_data.empty:
        raise ValueError(
            f"Équipement introuvable : {equipment_id}"
        )

    required_rows = int(
        duration / interval
    )

    if required_rows <= 0:
        raise ValueError(
            "La durée de l'anomalie doit être positive."
        )

    if required_rows > len(equipment_data):
        raise ValueError(
            f"Durée trop longue pour {equipment_id}."
        )

    target_timestamp = (
        equipment_data["timestamp"].min()
        + (
            equipment_data["timestamp"].max()
            - equipment_data["timestamp"].min()
        )
        * target_fraction
    )

    timestamp_values = (
        equipment_data["timestamp"]
        .to_numpy(dtype="datetime64[ns]")
    )

    target_position = int(
        np.abs(
            timestamp_values
            - np.datetime64(target_timestamp)
        ).argmin()
    )

    latest_start_position = (
        len(equipment_data)
        - required_rows
    )

    target_position = min(
        max(target_position, 0),
        latest_start_position,
    )

    maximum_search_steps = min(
        latest_start_position,
        int(pd.Timedelta(days=30) / interval),
    )

    for offset in build_candidate_offsets(
        maximum_search_steps
    ):
        start_position = target_position + offset

        if (
            start_position < 0
            or start_position > latest_start_position
        ):
            continue

        window = equipment_data.iloc[
            start_position:
            start_position + required_rows
        ]

        if len(window) != required_rows:
            continue

        if not (
            window["operating_state"] == "RUNNING"
        ).all():
            continue

        timestamp_differences = (
            window["timestamp"].diff().dropna()
        )

        if not (
            timestamp_differences == interval
        ).all():
            continue

        measurement_ids = set(
            window["measurement_id"].astype(str)
        )

        if (
            measurement_ids
            & occupied_measurement_ids
        ):
            continue

        return window.index

    raise RuntimeError(
        "Aucune fenêtre RUNNING continue n'a été "
        f"trouvée pour {equipment_id} pendant "
        f"{duration}."
    )


def recalculate_energy_columns(
    dataframe: pd.DataFrame,
    indexes: pd.Index,
    interval_hours: float,
) -> None:
    """Recalcule les colonnes dépendantes de la puissance."""

    dataframe.loc[
        indexes,
        "electrical_energy_kwh",
    ] = (
        dataframe.loc[
            indexes,
            "electrical_power_kw",
        ]
        * interval_hours
    )

    dataframe.loc[
        indexes,
        "current_ampere",
    ] = (
        dataframe.loc[
            indexes,
            "electrical_power_kw",
        ]
        / 6
    )

    production = dataframe.loc[
        indexes,
        "production_tonnes",
    ]

    energy = dataframe.loc[
        indexes,
        "electrical_energy_kwh",
    ]

    dataframe.loc[
        indexes,
        "specific_energy_kwh_t",
    ] = np.divide(
        energy,
        production,
        out=np.full(
            len(indexes),
            np.nan,
            dtype=float,
        ),
        where=production.to_numpy() > 0,
    )


def recalculate_production_columns(
    dataframe: pd.DataFrame,
    indexes: pd.Index,
    interval_hours: float,
    nominal_capacity_tph: float,
) -> None:
    """Recalcule les colonnes dépendantes du débit."""

    dataframe.loc[
        indexes,
        "production_tonnes",
    ] = (
        dataframe.loc[
            indexes,
            "production_rate_tph",
        ]
        * interval_hours
    )

    dataframe.loc[
        indexes,
        "capacity_utilization_pct",
    ] = (
        dataframe.loc[
            indexes,
            "production_rate_tph",
        ]
        / nominal_capacity_tph
        * 100
    )

    production = dataframe.loc[
        indexes,
        "production_tonnes",
    ]

    energy = dataframe.loc[
        indexes,
        "electrical_energy_kwh",
    ]

    dataframe.loc[
        indexes,
        "specific_energy_kwh_t",
    ] = np.divide(
        energy,
        production,
        out=np.full(
            len(indexes),
            np.nan,
            dtype=float,
        ),
        where=production.to_numpy() > 0,
    )


def register_anomaly(
    dataframe: pd.DataFrame,
    indexes: pd.Index,
    anomaly_id: str,
    anomaly_type: str,
    severity: str,
    injected_value_pct: float | None,
    description: str,
    parameters: dict[str, Any],
    anomaly_events: list[dict[str, Any]],
    anomaly_truth_rows: list[dict[str, Any]],
) -> None:
    """Enregistre l'événement et la vérité terrain ligne par ligne."""

    window = dataframe.loc[indexes].sort_values(
        "timestamp"
    )

    anomaly_events.append(
        {
            "anomaly_id": anomaly_id,
            "equipment_id": (
                window["equipment_id"].iloc[0]
            ),
            "start_time": (
                window["timestamp"].min()
            ),
            "end_time": (
                window["timestamp"].max()
            ),
            "anomaly_type": anomaly_type,
            "severity": severity,
            "affected_row_count": len(window),
            "injected_value_pct": (
                injected_value_pct
            ),
            "description": description,
            "parameters_json": json.dumps(
                parameters,
                ensure_ascii=False,
            ),
            "is_anomaly": 1,
        }
    )

    for row in window[
        [
            "measurement_id",
            "timestamp",
            "equipment_id",
        ]
    ].itertuples(index=False):
        anomaly_truth_rows.append(
            {
                "measurement_id": row.measurement_id,
                "timestamp": row.timestamp,
                "equipment_id": row.equipment_id,
                "is_anomaly": 1,
                "anomaly_id": anomaly_id,
                "anomaly_type": anomaly_type,
                "severity": severity,
            }
        )


def main() -> None:
    """Injecte les anomalies métier contrôlées."""

    print("Chargement des données normales...")

    config = load_yaml(CONFIG_PATH)
    dataframe = load_normal_data()
    equipment = load_equipment_reference()

    interval = get_interval(config)
    interval_hours = (
        interval.total_seconds() / 3600
    )

    nominal_capacity = (
        equipment
        .set_index("equipment_id")[
            "nominal_capacity_tph"
        ]
        .astype(float)
        .to_dict()
    )

    anomalous = dataframe.copy(deep=True)

    occupied_measurement_ids: set[str] = set()
    anomaly_events: list[dict[str, Any]] = []
    anomaly_truth_rows: list[dict[str, Any]] = []

    scenarios = [
        {
            "anomaly_id": "ANO_0001",
            "equipment_id": "KILN_001",
            "anomaly_type": "ENERGY_OVERCONSUMPTION",
            "severity": "HIGH",
            "target_fraction": 0.10,
            "duration": pd.Timedelta(hours=4),
            "injected_value_pct": 45.0,
            "description": (
                "Hausse synthétique de 45 % de la puissance "
                "électrique à production presque inchangée."
            ),
        },
        {
            "anomaly_id": "ANO_0002",
            "equipment_id": "CEMENT_MILL_001",
            "anomaly_type": "PRODUCTION_DROP",
            "severity": "HIGH",
            "target_fraction": 0.25,
            "duration": pd.Timedelta(hours=4),
            "injected_value_pct": -55.0,
            "description": (
                "Baisse synthétique de 55 % de la production "
                "avec consommation électrique maintenue."
            ),
        },
        {
            "anomaly_id": "ANO_0003",
            "equipment_id": "KILN_001",
            "anomaly_type": "OVERHEATING",
            "severity": "CRITICAL",
            "target_fraction": 0.40,
            "duration": pd.Timedelta(hours=5),
            "injected_value_pct": None,
            "description": (
                "Hausse synthétique de la température du four "
                "avec augmentation du nombre d'alarmes."
            ),
        },
        {
            "anomaly_id": "ANO_0004",
            "equipment_id": "CEMENT_MILL_001",
            "anomaly_type": "EXCESSIVE_VIBRATION",
            "severity": "HIGH",
            "target_fraction": 0.55,
            "duration": pd.Timedelta(hours=4),
            "injected_value_pct": 180.0,
            "description": (
                "Multiplication synthétique des vibrations "
                "par 2,8."
            ),
        },
        {
            "anomaly_id": "ANO_0005",
            "equipment_id": "COOLER_001",
            "anomaly_type": "PRESSURE_DROP",
            "severity": "MEDIUM",
            "target_fraction": 0.68,
            "duration": pd.Timedelta(hours=3),
            "injected_value_pct": -65.0,
            "description": (
                "Chute synthétique de 65 % de la pression."
            ),
        },
        {
            "anomaly_id": "ANO_0006",
            "equipment_id": "RAW_MILL_001",
            "anomaly_type": "SENSOR_STUCK",
            "severity": "MEDIUM",
            "target_fraction": 0.80,
            "duration": pd.Timedelta(hours=6),
            "injected_value_pct": None,
            "description": (
                "Capteur de température synthétiquement bloqué "
                "sur une valeur constante."
            ),
        },
        {
            "anomaly_id": "ANO_0007",
            "equipment_id": "KILN_001",
            "anomaly_type": "ENERGY_DRIFT",
            "severity": "HIGH",
            "target_fraction": 0.90,
            "duration": pd.Timedelta(hours=72),
            "injected_value_pct": 30.0,
            "description": (
                "Dérive progressive synthétique de la puissance "
                "électrique jusqu'à +30 %."
            ),
        },
    ]

    for scenario in scenarios:
        equipment_id = scenario["equipment_id"]

        print(
            f"Injection de {scenario['anomaly_type']} "
            f"sur {equipment_id}..."
        )

        indexes = find_running_window(
            dataframe=anomalous,
            equipment_id=equipment_id,
            target_fraction=float(
                scenario["target_fraction"]
            ),
            duration=scenario["duration"],
            interval=interval,
            occupied_measurement_ids=(
                occupied_measurement_ids
            ),
        )

        anomaly_type = scenario["anomaly_type"]
        parameters: dict[str, Any] = {}

        if anomaly_type == "ENERGY_OVERCONSUMPTION":
            factor = 1.45

            anomalous.loc[
                indexes,
                "electrical_power_kw",
            ] *= factor

            recalculate_energy_columns(
                anomalous,
                indexes,
                interval_hours,
            )

            parameters = {
                "power_factor": factor,
            }

        elif anomaly_type == "PRODUCTION_DROP":
            factor = 0.45

            anomalous.loc[
                indexes,
                "production_rate_tph",
            ] *= factor

            recalculate_production_columns(
                anomalous,
                indexes,
                interval_hours,
                nominal_capacity[equipment_id],
            )

            parameters = {
                "production_factor": factor,
            }

        elif anomaly_type == "OVERHEATING":
            temperature_increase_c = 130.0

            anomalous.loc[
                indexes,
                "temperature_c",
            ] += temperature_increase_c

            anomalous.loc[
                indexes,
                "alarm_count",
            ] += 2

            parameters = {
                "temperature_increase_c": (
                    temperature_increase_c
                ),
                "additional_alarms": 2,
            }

        elif anomaly_type == "EXCESSIVE_VIBRATION":
            factor = 2.8

            anomalous.loc[
                indexes,
                "vibration_mm_s",
            ] *= factor

            anomalous.loc[
                indexes,
                "alarm_count",
            ] += 1

            parameters = {
                "vibration_factor": factor,
                "additional_alarms": 1,
            }

        elif anomaly_type == "PRESSURE_DROP":
            factor = 0.35

            anomalous.loc[
                indexes,
                "pressure_bar",
            ] *= factor

            anomalous.loc[
                indexes,
                "alarm_count",
            ] += 1

            parameters = {
                "pressure_factor": factor,
                "additional_alarms": 1,
            }

        elif anomaly_type == "SENSOR_STUCK":
            original_values = anomalous.loc[
                indexes,
                "temperature_c",
            ]

            stuck_value = round(
                float(original_values.mean()) + 0.137,
                3,
            )

            anomalous.loc[
                indexes,
                "temperature_c",
            ] = stuck_value

            parameters = {
                "sensor": "temperature_c",
                "stuck_value": stuck_value,
            }

        elif anomaly_type == "ENERGY_DRIFT":
            factors = np.linspace(
                1.0,
                1.30,
                num=len(indexes),
            )

            anomalous.loc[
                indexes,
                "electrical_power_kw",
            ] = (
                anomalous.loc[
                    indexes,
                    "electrical_power_kw",
                ].to_numpy()
                * factors
            )

            recalculate_energy_columns(
                anomalous,
                indexes,
                interval_hours,
            )

            parameters = {
                "start_factor": 1.0,
                "end_factor": 1.30,
            }

        else:
            raise ValueError(
                f"Type d'anomalie non géré : "
                f"{anomaly_type}"
            )

        register_anomaly(
            dataframe=anomalous,
            indexes=indexes,
            anomaly_id=scenario["anomaly_id"],
            anomaly_type=anomaly_type,
            severity=scenario["severity"],
            injected_value_pct=(
                scenario["injected_value_pct"]
            ),
            description=scenario["description"],
            parameters=parameters,
            anomaly_events=anomaly_events,
            anomaly_truth_rows=(
                anomaly_truth_rows
            ),
        )

        occupied_measurement_ids.update(
            anomalous.loc[
                indexes,
                "measurement_id",
            ].astype(str)
        )

    # Arrondir d'abord les mesures de base.
    # Les colonnes calculées seront ensuite recalculées à partir
    # des valeurs arrondies afin de préserver exactement les
    # relations mathématiques du jeu de données.
    numeric_columns = anomalous.select_dtypes(
        include=["number"]
    ).columns

    anomalous[numeric_columns] = (
        anomalous[numeric_columns]
        .round(3)
    )

    anomaly_mask = (
        anomalous["measurement_id"]
        .astype(str)
        .isin(occupied_measurement_ids)
    )

    anomaly_indexes = anomalous.index[
        anomaly_mask
    ]

    if len(anomaly_indexes) == 0:
        raise ValueError(
            "Aucune ligne anormale n'a été retrouvée "
            "avant l'enregistrement."
        )

    # Recalcul de la production sur cinq minutes à partir
    # du débit arrondi.
    anomalous.loc[
        anomaly_indexes,
        "production_tonnes",
    ] = (
        anomalous.loc[
            anomaly_indexes,
            "production_rate_tph",
        ]
        * interval_hours
    ).round(3)

    # Recalcul du taux d'utilisation de la capacité.
    anomaly_capacities = (
        anomalous.loc[
            anomaly_indexes,
            "equipment_id",
        ]
        .map(nominal_capacity)
        .astype(float)
    )

    if anomaly_capacities.isna().any():
        unknown_equipment = (
            anomalous.loc[
                anomaly_indexes[
                    anomaly_capacities.isna()
                ],
                "equipment_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Capacité nominale absente pour : "
            f"{unknown_equipment}"
        )

    anomalous.loc[
        anomaly_indexes,
        "capacity_utilization_pct",
    ] = (
        anomalous.loc[
            anomaly_indexes,
            "production_rate_tph",
        ].to_numpy()
        / anomaly_capacities.to_numpy()
        * 100
    ).round(3)

    # Recalcul de l'énergie consommée à partir de la
    # puissance électrique arrondie.
    anomalous.loc[
        anomaly_indexes,
        "electrical_energy_kwh",
    ] = (
        anomalous.loc[
            anomaly_indexes,
            "electrical_power_kw",
        ]
        * interval_hours
    ).round(3)

    # Recalcul cohérent de l'intensité synthétique.
    anomalous.loc[
        anomaly_indexes,
        "current_ampere",
    ] = (
        anomalous.loc[
            anomaly_indexes,
            "electrical_power_kw",
        ]
        / 6
    ).round(3)

    # Recalcul final de la consommation spécifique à partir
    # des valeurs effectivement enregistrées.
    anomaly_production = anomalous.loc[
        anomaly_indexes,
        "production_tonnes",
    ].to_numpy(dtype=float)

    anomaly_energy = anomalous.loc[
        anomaly_indexes,
        "electrical_energy_kwh",
    ].to_numpy(dtype=float)

    anomaly_specific_energy = np.divide(
        anomaly_energy,
        anomaly_production,
        out=np.full(
            len(anomaly_indexes),
            np.nan,
            dtype=float,
        ),
        where=anomaly_production > 0,
    )

    anomalous.loc[
        anomaly_indexes,
        "specific_energy_kwh_t",
    ] = np.round(
        anomaly_specific_energy,
        3,
    )

    anomaly_events_dataframe = pd.DataFrame(
        anomaly_events
    )

    anomaly_truth_dataframe = pd.DataFrame(
        anomaly_truth_rows
    ).sort_values(
        ["timestamp", "equipment_id"]
    )

    if (
        anomaly_truth_dataframe["measurement_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Certaines anomalies se chevauchent."
        )

    LABELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANOMALOUS_PARQUET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomalous.to_parquet(
        ANOMALOUS_PARQUET_PATH,
        index=False,
    )

    anomalous.to_csv(
        ANOMALOUS_CSV_PATH,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    anomaly_events_dataframe.to_csv(
        ANOMALY_LABELS_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    anomaly_truth_dataframe.to_parquet(
        ANOMALY_TRUTH_PATH,
        index=False,
    )

    print()
    print("Injection terminée avec succès.")
    print(
        f"Nombre d'événements injectés : "
        f"{len(anomaly_events_dataframe)}"
    )
    print(
        f"Nombre de lignes étiquetées : "
        f"{len(anomaly_truth_dataframe):,}"
    )
    print(
        f"Données avec anomalies : "
        f"{ANOMALOUS_PARQUET_PATH}"
    )
    print(
        f"Événements : {ANOMALY_LABELS_PATH}"
    )
    print(
        f"Vérité terrain : {ANOMALY_TRUTH_PATH}"
    )


if __name__ == "__main__":
    main()