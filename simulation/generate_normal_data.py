from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.yml"
EQUIPMENT_PATH = PROJECT_ROOT / "data" / "reference" / "equipment.csv"
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"


def load_config() -> dict[str, Any]:
    """Charge le fichier YAML de configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Le fichier de configuration est vide.")

    return config


def load_equipment() -> pd.DataFrame:
    """Charge la liste des équipements."""
    if not EQUIPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier des équipements introuvable : {EQUIPMENT_PATH}"
        )

    equipment = pd.read_csv(EQUIPMENT_PATH)

    if equipment.empty:
        raise ValueError("Le fichier equipment.csv est vide.")

    required_columns = {
        "equipment_id",
        "equipment_name",
        "equipment_type",
        "site_id",
        "workshop_id",
        "nominal_capacity_tph",
        "installation_date",
        "status",
    }

    missing_columns = required_columns - set(equipment.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans equipment.csv : {sorted(missing_columns)}"
        )

    return equipment


def determine_shift(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Attribue l'équipe de travail selon l'heure :
    SHIFT_1 : 06:00 - 13:59
    SHIFT_2 : 14:00 - 21:59
    SHIFT_3 : 22:00 - 05:59
    """
    hours = timestamps.hour.to_numpy()

    return np.select(
        condlist=[
            (hours >= 6) & (hours < 14),
            (hours >= 14) & (hours < 22),
        ],
        choicelist=[
            "SHIFT_1",
            "SHIFT_2",
        ],
        default="SHIFT_3",
    )


def create_planned_maintenance_periods(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Crée une maintenance planifiée le premier dimanche
    de chaque mois entre 02:00 et 04:00.
    """
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    first_month = start_date.normalize().replace(day=1)
    month_starts = pd.date_range(
        start=first_month,
        end=end_date.normalize(),
        freq="MS",
    )

    for month_start in month_starts:
        days_until_sunday = (6 - month_start.dayofweek) % 7

        maintenance_start = (
            month_start
            + pd.Timedelta(days=days_until_sunday)
            + pd.Timedelta(hours=2)
        )
        maintenance_end = maintenance_start + pd.Timedelta(hours=2)

        if maintenance_start <= end_date and maintenance_end >= start_date:
            periods.append((maintenance_start, maintenance_end))

    return periods


def build_maintenance_mask(
    timestamps: pd.DatetimeIndex,
    maintenance_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    """Retourne True pour les périodes de maintenance."""
    mask = np.zeros(len(timestamps), dtype=bool)

    for maintenance_start, maintenance_end in maintenance_periods:
        mask |= np.asarray(
            (timestamps >= maintenance_start)
            & (timestamps < maintenance_end)
        )

    return mask


def generate_equipment_measurements(
    equipment_row: pd.Series,
    equipment_config: dict[str, Any],
    timestamps: pd.DatetimeIndex,
    interval_hours: float,
    rng: np.random.Generator,
    ingestion_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Génère les mesures normales d'un seul équipement."""
    equipment_id = str(equipment_row["equipment_id"])
    equipment_type = str(equipment_row["equipment_type"])
    nominal_capacity = float(equipment_row["nominal_capacity_tph"])
    number_of_rows = len(timestamps)

    maintenance_periods = create_planned_maintenance_periods(
        start_date=timestamps.min(),
        end_date=timestamps.max(),
    )
    maintenance_mask = build_maintenance_mask(
        timestamps=timestamps,
        maintenance_periods=maintenance_periods,
    )

    operating_state = np.where(
        maintenance_mask,
        "MAINTENANCE",
        "RUNNING",
    )

    shifts = determine_shift(timestamps)

    shift_factor = np.select(
        condlist=[
            shifts == "SHIFT_1",
            shifts == "SHIFT_2",
            shifts == "SHIFT_3",
        ],
        choicelist=[
            1.00,
            0.98,
            0.95,
        ],
        default=1.00,
    ).astype(float)

    hour_decimal = (
        timestamps.hour.to_numpy(dtype=float)
        + timestamps.minute.to_numpy(dtype=float) / 60.0
    )

    daily_factor = np.asarray(
        1.0 + 0.02 * np.sin(2.0 * np.pi * hour_decimal / 24.0),
        dtype=float,
    )

    utilization_min = float(
        equipment_config["normal_utilization_min"]
    )
    utilization_max = float(
        equipment_config["normal_utilization_max"]
    )

    utilization = rng.uniform(
        low=utilization_min,
        high=utilization_max,
        size=number_of_rows,
    )

    utilization = np.asarray(
        utilization
        * shift_factor
        * daily_factor
        + rng.normal(
            loc=0.0,
            scale=0.01,
            size=number_of_rows,
        ),
        dtype=float,
    )

    utilization = np.clip(utilization, 0.50, 1.00)
    utilization[maintenance_mask] = 0.0

    production_rate_tph = nominal_capacity * utilization
    production_tonnes = production_rate_tph * interval_hours
    capacity_utilization_pct = utilization * 100.0

    base_power_kw = float(equipment_config["base_power_kw"])
    production_power_coefficient = float(
        equipment_config["production_power_coefficient"]
    )

    power_noise = rng.normal(
        loc=0.0,
        scale=base_power_kw * 0.015,
        size=number_of_rows,
    )

    electrical_power_kw = (
        base_power_kw
        + production_power_coefficient * production_rate_tph
        + power_noise
    )

    maintenance_count = int(maintenance_mask.sum())
    if maintenance_count > 0:
        electrical_power_kw[maintenance_mask] = (
            base_power_kw * 0.05
            + np.abs(
                rng.normal(
                    loc=0.0,
                    scale=base_power_kw * 0.005,
                    size=maintenance_count,
                )
            )
        )

    electrical_power_kw = np.clip(
        electrical_power_kw,
        0.0,
        None,
    )
    electrical_energy_kwh = electrical_power_kw * interval_hours

    specific_energy_kwh_t = np.divide(
        electrical_energy_kwh,
        production_tonnes,
        out=np.full(number_of_rows, np.nan, dtype=float),
        where=production_tonnes > 0.0,
    )

    load_ratio = np.divide(
        production_rate_tph,
        nominal_capacity,
        out=np.zeros(number_of_rows, dtype=float),
        where=nominal_capacity > 0.0,
    )

    temperature_mean = float(
        equipment_config["temperature"]["mean"]
    )
    temperature_std = float(
        equipment_config["temperature"]["standard_deviation"]
    )

    temperature_c = (
        temperature_mean
        + (load_ratio - 0.80) * temperature_std * 0.80
        + rng.normal(
            loc=0.0,
            scale=temperature_std,
            size=number_of_rows,
        )
    )

    if maintenance_count > 0:
        temperature_c[maintenance_mask] = (
            temperature_mean * 0.50
            + rng.normal(
                loc=0.0,
                scale=max(1.0, temperature_std * 0.30),
                size=maintenance_count,
            )
        )

    temperature_c = np.clip(temperature_c, 0.0, None)

    pressure_mean = float(
        equipment_config["pressure"]["mean"]
    )
    pressure_std = float(
        equipment_config["pressure"]["standard_deviation"]
    )

    pressure_bar = (
        pressure_mean
        + (load_ratio - 0.80) * pressure_mean * 0.10
        + rng.normal(
            loc=0.0,
            scale=pressure_std,
            size=number_of_rows,
        )
    )

    if maintenance_count > 0:
        pressure_bar[maintenance_mask] = np.maximum(
            0.0,
            pressure_mean * 0.10
            + rng.normal(
                loc=0.0,
                scale=max(0.01, pressure_std * 0.20),
                size=maintenance_count,
            ),
        )

    pressure_bar = np.clip(pressure_bar, 0.0, None)

    vibration_mean = float(
        equipment_config["vibration"]["mean"]
    )
    vibration_std = float(
        equipment_config["vibration"]["standard_deviation"]
    )

    vibration_mm_s = (
        vibration_mean
        + (load_ratio - 0.80) * 0.80
        + rng.normal(
            loc=0.0,
            scale=vibration_std,
            size=number_of_rows,
        )
    )

    if maintenance_count > 0:
        vibration_mm_s[maintenance_mask] = np.maximum(
            0.0,
            rng.normal(
                loc=0.10,
                scale=0.03,
                size=maintenance_count,
            ),
        )

    vibration_mm_s = np.clip(vibration_mm_s, 0.0, None)

    rotation_mean = float(
        equipment_config["rotation_speed"]["mean"]
    )
    rotation_std = float(
        equipment_config["rotation_speed"]["standard_deviation"]
    )

    rotation_speed_rpm = (
        rotation_mean * (0.50 + 0.60 * load_ratio)
        + rng.normal(
            loc=0.0,
            scale=rotation_std,
            size=number_of_rows,
        )
    )

    rotation_speed_rpm[maintenance_mask] = 0.0
    rotation_speed_rpm = np.clip(
        rotation_speed_rpm,
        0.0,
        None,
    )

    current_ampere = (
        electrical_power_kw / 6.0
        + rng.normal(
            loc=0.0,
            scale=5.0,
            size=number_of_rows,
        )
    )
    current_ampere = np.clip(current_ampere, 0.0, None)

    if maintenance_count > 0:
        current_ampere[maintenance_mask] = (
            electrical_power_kw[maintenance_mask] / 6.0
        )

    alarm_count = rng.poisson(
        lam=0.01,
        size=number_of_rows,
    )
    alarm_count[maintenance_mask] = 0

    if equipment_type == "rotary_kiln":
        thermal_specific_mj_t = rng.normal(
            loc=3200.0,
            scale=45.0,
            size=number_of_rows,
        )
        thermal_energy_mj = (
            production_tonnes * thermal_specific_mj_t
        )
    else:
        thermal_energy_mj = np.zeros(
            number_of_rows,
            dtype=float,
        )

    thermal_energy_mj[maintenance_mask] = 0.0

    measurement_ids = [
        f"MES_{equipment_id}_{index:07d}"
        for index in range(1, number_of_rows + 1)
    ]

    measurements = pd.DataFrame(
        {
            "measurement_id": measurement_ids,
            "timestamp": timestamps,
            "site_id": str(equipment_row["site_id"]),
            "workshop_id": str(equipment_row["workshop_id"]),
            "equipment_id": equipment_id,
            "shift": shifts,
            "operating_state": operating_state,
            "production_rate_tph": production_rate_tph,
            "production_tonnes": production_tonnes,
            "capacity_utilization_pct": capacity_utilization_pct,
            "electrical_power_kw": electrical_power_kw,
            "electrical_energy_kwh": electrical_energy_kwh,
            "thermal_energy_mj": thermal_energy_mj,
            "specific_energy_kwh_t": specific_energy_kwh_t,
            "temperature_c": temperature_c,
            "pressure_bar": pressure_bar,
            "vibration_mm_s": vibration_mm_s,
            "current_ampere": current_ampere,
            "rotation_speed_rpm": rotation_speed_rpm,
            "alarm_count": alarm_count,
            "source_system": "SIMULATOR_V1",
            "ingestion_timestamp": ingestion_timestamp,
            "data_quality_status": "VALID",
            "has_missing_value": False,
            "is_duplicate": False,
        }
    )

    downtime_events: list[dict[str, Any]] = []

    for maintenance_start, maintenance_end in maintenance_periods:
        duration_minutes = int(
            (maintenance_end - maintenance_start).total_seconds()
            / 60.0
        )

        downtime_events.append(
            {
                "equipment_id": equipment_id,
                "start_time": maintenance_start,
                "end_time": maintenance_end,
                "duration_minutes": duration_minutes,
                "downtime_type": "PLANNED",
                "reason_code": "PREVENTIVE_MAINTENANCE",
                "severity": "LOW",
                "description": "Maintenance planifiée synthétique",
            }
        )

    return measurements, downtime_events


def round_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Arrondit les colonnes numériques pour faciliter la lecture."""
    decimals = {
        "production_rate_tph": 3,
        "production_tonnes": 3,
        "capacity_utilization_pct": 3,
        "electrical_power_kw": 3,
        "electrical_energy_kwh": 3,
        "thermal_energy_mj": 3,
        "specific_energy_kwh_t": 3,
        "temperature_c": 3,
        "pressure_bar": 3,
        "vibration_mm_s": 3,
        "current_ampere": 3,
        "rotation_speed_rpm": 3,
    }

    return dataframe.round(decimals)


def main() -> None:
    """Lance la génération complète des données normales."""
    print("Chargement de la configuration...")

    config = load_config()
    equipment = load_equipment()
    simulation_config = config["simulation"]

    start_date = pd.Timestamp(
        simulation_config["start_date"]
    )
    end_date = pd.Timestamp(
        simulation_config["end_date"]
    )
    frequency = str(
        simulation_config["frequency"]
    )
    random_seed = int(
        simulation_config["random_seed"]
    )

    if end_date < start_date:
        raise ValueError(
            "La date de fin doit être postérieure à la date de début."
        )

    interval = pd.Timedelta(frequency)
    interval_hours = interval.total_seconds() / 3600.0

    if interval_hours <= 0.0:
        raise ValueError(
            "La fréquence doit être strictement positive."
        )

    print("Création des dates de mesure...")

    timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        freq=frequency,
    )

    if len(timestamps) == 0:
        raise ValueError(
            "Aucune date de mesure n'a été générée."
        )

    print(
        f"Nombre de dates par équipement : {len(timestamps):,}"
    )

    rng = np.random.default_rng(seed=random_seed)
    ingestion_timestamp = end_date + pd.Timedelta(days=1)

    all_measurements: list[pd.DataFrame] = []
    all_downtime_events: list[dict[str, Any]] = []

    for _, equipment_row in equipment.iterrows():
        equipment_id = str(equipment_row["equipment_id"])

        print(
            f"Génération des mesures pour {equipment_id}..."
        )

        if equipment_id not in config["equipment"]:
            raise KeyError(
                f"L'équipement {equipment_id} "
                "n'existe pas dans config/simulation.yml."
            )

        measurements, downtime_events = (
            generate_equipment_measurements(
                equipment_row=equipment_row,
                equipment_config=config["equipment"][
                    equipment_id
                ],
                timestamps=timestamps,
                interval_hours=interval_hours,
                rng=rng,
                ingestion_timestamp=ingestion_timestamp,
            )
        )

        all_measurements.append(measurements)
        all_downtime_events.extend(downtime_events)

    print("Assemblage des données...")

    final_measurements = pd.concat(
        all_measurements,
        ignore_index=True,
    )
    final_measurements = round_numeric_columns(
        final_measurements
    )

    downtime_dataframe = pd.DataFrame(
        all_downtime_events
    )

    if not downtime_dataframe.empty:
        downtime_dataframe.insert(
            0,
            "downtime_id",
            [
                f"STOP_{index:05d}"
                for index in range(
                    1,
                    len(downtime_dataframe) + 1,
                )
            ],
        )

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements_csv_path = (
        RAW_DIRECTORY
        / "industrial_measurements_normal.csv"
    )
    measurements_parquet_path = (
        RAW_DIRECTORY
        / "industrial_measurements_normal.parquet"
    )
    downtime_csv_path = (
        RAW_DIRECTORY
        / "downtime_events_normal.csv"
    )

    final_measurements.to_csv(
        measurements_csv_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    final_measurements.to_parquet(
        measurements_parquet_path,
        index=False,
    )

    downtime_dataframe.to_csv(
        downtime_csv_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print()
    print("Génération terminée avec succès.")
    print(
        f"Nombre total de mesures : {len(final_measurements):,}"
    )
    print(
        f"Nombre d'arrêts planifiés : {len(downtime_dataframe):,}"
    )
    print()
    print(f"CSV : {measurements_csv_path}")
    print(f"Parquet : {measurements_parquet_path}")
    print(f"Arrêts : {downtime_csv_path}")


if __name__ == "__main__":
    main()