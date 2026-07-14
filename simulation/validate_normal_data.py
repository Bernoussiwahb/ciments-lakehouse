from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.yml"

MEASUREMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_normal.parquet"
)

DOWNTIME_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "downtime_events_normal.csv"
)


def load_config() -> dict:
    """Charge le fichier de configuration YAML."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Le fichier de configuration est vide.")

    return config


def main() -> None:
    """Valide les données industrielles normales générées."""

    if not MEASUREMENTS_PATH.exists():
        raise FileNotFoundError(
            "Le fichier des mesures n'existe pas : "
            f"{MEASUREMENTS_PATH}\n"
            "Exécute d'abord simulation/generate_normal_data.py."
        )

    if not DOWNTIME_PATH.exists():
        raise FileNotFoundError(
            "Le fichier des arrêts n'existe pas : "
            f"{DOWNTIME_PATH}\n"
            "Exécute d'abord simulation/generate_normal_data.py."
        )

    measurements = pd.read_parquet(MEASUREMENTS_PATH)

    downtime = pd.read_csv(
        DOWNTIME_PATH,
        parse_dates=["start_time", "end_time"],
    )

    config = load_config()

    required_columns = {
        "measurement_id",
        "timestamp",
        "equipment_id",
        "operating_state",
        "production_rate_tph",
        "production_tonnes",
        "electrical_power_kw",
        "electrical_energy_kwh",
        "thermal_energy_mj",
        "specific_energy_kwh_t",
        "temperature_c",
        "pressure_bar",
        "vibration_mm_s",
        "current_ampere",
        "rotation_speed_rpm",
        "alarm_count",
        "data_quality_status",
    }

    missing_columns = required_columns - set(measurements.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    if measurements.empty:
        raise ValueError("Le fichier des mesures est vide.")

    if measurements["measurement_id"].duplicated().any():
        raise ValueError("Des measurement_id sont dupliqués.")

    duplicated_measurements = measurements.duplicated(
        subset=["equipment_id", "timestamp"]
    )

    if duplicated_measurements.any():
        raise ValueError(
            "Un équipement possède plusieurs mesures au même timestamp."
        )

    null_columns = measurements.columns[
        measurements.isna().any()
    ].tolist()

    unexpected_null_columns = [
        column
        for column in null_columns
        if column != "specific_energy_kwh_t"
    ]

    if unexpected_null_columns:
        raise ValueError(
            "Valeurs manquantes inattendues dans : "
            f"{unexpected_null_columns}"
        )

    non_negative_columns = [
        "production_rate_tph",
        "production_tonnes",
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

    for column in non_negative_columns:
        if (measurements[column] < 0).any():
            raise ValueError(
                f"Valeurs négatives détectées dans {column}."
            )

    maintenance_rows = measurements[
        measurements["operating_state"] == "MAINTENANCE"
    ]

    if maintenance_rows.empty:
        raise ValueError(
            "Aucune période de maintenance n'a été trouvée."
        )

    if not np.allclose(
        maintenance_rows["production_rate_tph"],
        0,
        atol=0.001,
    ):
        raise ValueError(
            "La production doit être nulle pendant la maintenance."
        )

    if not np.allclose(
        maintenance_rows["production_tonnes"],
        0,
        atol=0.001,
    ):
        raise ValueError(
            "La quantité produite doit être nulle pendant la maintenance."
        )

    running_rows = measurements[
        measurements["operating_state"] == "RUNNING"
    ]

    if running_rows.empty:
        raise ValueError(
            "Aucune mesure RUNNING n'a été trouvée."
        )

    if (running_rows["production_rate_tph"] <= 0).any():
        raise ValueError(
            "Une mesure RUNNING contient une production nulle ou négative."
        )

    interval_hours = (
        pd.Timedelta(
            config["simulation"]["frequency"]
        ).total_seconds()
        / 3600
    )

    expected_production = (
        measurements["production_rate_tph"]
        * interval_hours
    )

    if not np.allclose(
        measurements["production_tonnes"],
        expected_production,
        atol=0.01,
    ):
        raise ValueError(
            "La formule production_tonnes = "
            "production_rate_tph × durée n'est pas respectée."
        )

    expected_energy = (
        measurements["electrical_power_kw"]
        * interval_hours
    )

    if not np.allclose(
        measurements["electrical_energy_kwh"],
        expected_energy,
        atol=0.01,
    ):
        raise ValueError(
            "La formule electrical_energy_kwh = "
            "electrical_power_kw × durée n'est pas respectée."
        )

    equipment_count = len(config["equipment"])

    timestamps = pd.date_range(
        start=config["simulation"]["start_date"],
        end=config["simulation"]["end_date"],
        freq=config["simulation"]["frequency"],
    )

    expected_rows = len(timestamps) * equipment_count

    if len(measurements) != expected_rows:
        raise ValueError(
            f"Nombre de lignes incorrect : "
            f"{len(measurements):,} au lieu de {expected_rows:,}."
        )

    if not (measurements["data_quality_status"] == "VALID").all():
        raise ValueError(
            "Certaines données normales ne sont pas marquées VALID."
        )

    print("Validation réussie.")
    print()
    print(f"Nombre de mesures : {len(measurements):,}")
    print(
        "Nombre d'équipements : "
        f"{measurements['equipment_id'].nunique()}"
    )
    print(f"Nombre de maintenances : {len(downtime):,}")
    print("Valeurs manquantes inattendues : 0")
    print("Doublons équipement/timestamp : 0")
    print()

    summary = (
        measurements
        .groupby("equipment_id")
        .agg(
            number_of_measurements=(
                "measurement_id",
                "count",
            ),
            average_production_tph=(
                "production_rate_tph",
                "mean",
            ),
            average_power_kw=(
                "electrical_power_kw",
                "mean",
            ),
            maintenance_rows=(
                "operating_state",
                lambda values: (
                    values == "MAINTENANCE"
                ).sum(),
            ),
        )
        .round(2)
    )

    print("Résumé par équipement :")
    print(summary.to_string())


if __name__ == "__main__":
    main()