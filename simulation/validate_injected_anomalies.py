from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.yml"

NORMAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_normal.parquet"
)

ANOMALOUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_with_anomalies.parquet"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "anomaly_labels.csv"
)

TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "anomaly_truth.parquet"
)


def load_config() -> dict:
    """Charge la configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            "Le fichier de configuration est vide."
        )

    return config


def assert_files_exist() -> None:
    """Vérifie la présence de tous les fichiers."""

    required_paths = [
        NORMAL_PATH,
        ANOMALOUS_PATH,
        LABELS_PATH,
        TRUTH_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Fichiers manquants :\n"
            f"{formatted}\n"
            "Exécute d'abord inject_anomalies.py."
        )


def main() -> None:
    """Valide les anomalies injectées."""

    assert_files_exist()

    normal = pd.read_parquet(NORMAL_PATH)
    anomalous = pd.read_parquet(ANOMALOUS_PATH)

    labels = pd.read_csv(
        LABELS_PATH,
        parse_dates=[
            "start_time",
            "end_time",
        ],
    )

    truth = pd.read_parquet(TRUTH_PATH)

    normal["timestamp"] = pd.to_datetime(
        normal["timestamp"]
    )
    anomalous["timestamp"] = pd.to_datetime(
        anomalous["timestamp"]
    )
    truth["timestamp"] = pd.to_datetime(
        truth["timestamp"]
    )

    if len(normal) != len(anomalous):
        raise ValueError(
            "Le nombre de lignes a changé après injection."
        )

    if not normal["measurement_id"].equals(
        anomalous["measurement_id"]
    ):
        raise ValueError(
            "Les measurement_id ou leur ordre ont changé."
        )

    if len(labels) != 7:
        raise ValueError(
            f"Sept événements étaient attendus, "
            f"mais {len(labels)} ont été trouvés."
        )

    expected_types = {
        "ENERGY_OVERCONSUMPTION",
        "PRODUCTION_DROP",
        "OVERHEATING",
        "EXCESSIVE_VIBRATION",
        "PRESSURE_DROP",
        "SENSOR_STUCK",
        "ENERGY_DRIFT",
    }

    actual_types = set(
        labels["anomaly_type"]
    )

    if actual_types != expected_types:
        raise ValueError(
            "Types d'anomalies incorrects.\n"
            f"Attendus : {sorted(expected_types)}\n"
            f"Trouvés : {sorted(actual_types)}"
        )

    if truth["measurement_id"].duplicated().any():
        raise ValueError(
            "La vérité terrain contient des doublons."
        )

    if not set(
        truth["measurement_id"]
    ).issubset(
        set(anomalous["measurement_id"])
    ):
        raise ValueError(
            "La vérité terrain contient des identifiants "
            "inconnus."
        )

    comparison_columns = [
        "production_rate_tph",
        "production_tonnes",
        "capacity_utilization_pct",
        "electrical_power_kw",
        "electrical_energy_kwh",
        "specific_energy_kwh_t",
        "temperature_c",
        "pressure_bar",
        "vibration_mm_s",
        "current_ampere",
        "alarm_count",
    ]

    changed_mask = np.zeros(
        len(normal),
        dtype=bool,
    )

    for column in comparison_columns:
        normal_values = normal[column].to_numpy(
            dtype=float
        )
        anomalous_values = anomalous[
            column
        ].to_numpy(dtype=float)

        equal_values = np.isclose(
            normal_values,
            anomalous_values,
            rtol=0,
            atol=0.0001,
            equal_nan=True,
        )

        changed_mask |= ~equal_values

    changed_ids = set(
        anomalous.loc[
            changed_mask,
            "measurement_id",
        ]
    )

    truth_ids = set(
        truth["measurement_id"]
    )

    changed_outside_truth = (
        changed_ids - truth_ids
    )

    if changed_outside_truth:
        raise ValueError(
            "Des lignes ont été modifiées sans étiquette."
        )

    changed_coverage = (
        len(changed_ids) / len(truth_ids)
    )

    if changed_coverage < 0.95:
        raise ValueError(
            "Trop de lignes étiquetées ne présentent "
            "aucune modification réelle."
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
        if (anomalous[column] < 0).any():
            raise ValueError(
                f"Valeurs négatives dans {column}."
            )

    config = load_config()

    interval_hours = (
        pd.Timedelta(
            config["simulation"]["frequency"]
        ).total_seconds()
        / 3600
    )

    expected_energy = (
        anomalous["electrical_power_kw"]
        * interval_hours
    )

    if not np.allclose(
        anomalous["electrical_energy_kwh"],
        expected_energy,
        atol=0.01,
    ):
        raise ValueError(
            "La relation puissance/énergie "
            "n'est plus respectée."
        )

    expected_production = (
        anomalous["production_rate_tph"]
        * interval_hours
    )

    if not np.allclose(
        anomalous["production_tonnes"],
        expected_production,
        atol=0.01,
    ):
        raise ValueError(
            "La relation débit/production "
            "n'est plus respectée."
        )

    positive_production = (
        anomalous["production_tonnes"] > 0
    )

    expected_specific_energy = (
        anomalous.loc[
            positive_production,
            "electrical_energy_kwh",
        ]
        / anomalous.loc[
            positive_production,
            "production_tonnes",
        ]
    )

    if not np.allclose(
        anomalous.loc[
            positive_production,
            "specific_energy_kwh_t",
        ],
        expected_specific_energy,
        atol=0.01,
    ):
        raise ValueError(
            "La consommation spécifique "
            "n'est plus cohérente."
        )

    normal_indexed = normal.set_index(
        "measurement_id"
    )
    anomalous_indexed = anomalous.set_index(
        "measurement_id"
    )

    validation_results: list[str] = []

    for anomaly_type, truth_subset in truth.groupby(
        "anomaly_type"
    ):
        ids = truth_subset[
            "measurement_id"
        ].tolist()

        before = normal_indexed.loc[ids]
        after = anomalous_indexed.loc[ids]

        if anomaly_type == "ENERGY_OVERCONSUMPTION":
            ratio = (
                after["electrical_power_kw"].mean()
                / before["electrical_power_kw"].mean()
            )

            if ratio < 1.40:
                raise ValueError(
                    "Surconsommation énergétique trop faible."
                )

            validation_results.append(
                f"{anomaly_type}: puissance × {ratio:.2f}"
            )

        elif anomaly_type == "PRODUCTION_DROP":
            ratio = (
                after["production_rate_tph"].mean()
                / before["production_rate_tph"].mean()
            )

            if ratio > 0.50:
                raise ValueError(
                    "Baisse de production insuffisante."
                )

            validation_results.append(
                f"{anomaly_type}: production × {ratio:.2f}"
            )

        elif anomaly_type == "OVERHEATING":
            difference = (
                after["temperature_c"].mean()
                - before["temperature_c"].mean()
            )

            if difference < 120:
                raise ValueError(
                    "Surchauffe insuffisante."
                )

            validation_results.append(
                f"{anomaly_type}: +{difference:.1f} °C"
            )

        elif anomaly_type == "EXCESSIVE_VIBRATION":
            ratio = (
                after["vibration_mm_s"].mean()
                / before["vibration_mm_s"].mean()
            )

            if ratio < 2.70:
                raise ValueError(
                    "Hausse de vibration insuffisante."
                )

            validation_results.append(
                f"{anomaly_type}: vibration × {ratio:.2f}"
            )

        elif anomaly_type == "PRESSURE_DROP":
            ratio = (
                after["pressure_bar"].mean()
                / before["pressure_bar"].mean()
            )

            if ratio > 0.40:
                raise ValueError(
                    "Chute de pression insuffisante."
                )

            validation_results.append(
                f"{anomaly_type}: pression × {ratio:.2f}"
            )

        elif anomaly_type == "SENSOR_STUCK":
            standard_deviation = (
                after["temperature_c"].std()
            )

            if standard_deviation > 0.0001:
                raise ValueError(
                    "Le capteur simulé n'est pas bloqué."
                )

            validation_results.append(
                f"{anomaly_type}: écart-type = "
                f"{standard_deviation:.4f}"
            )

        elif anomaly_type == "ENERGY_DRIFT":
            count = len(after)
            quarter = max(1, count // 4)

            first_mean = (
                after["electrical_power_kw"]
                .iloc[:quarter]
                .mean()
            )

            last_mean = (
                after["electrical_power_kw"]
                .iloc[-quarter:]
                .mean()
            )

            normal_first_mean = (
                before["electrical_power_kw"]
                .iloc[:quarter]
                .mean()
            )

            normal_last_mean = (
                before["electrical_power_kw"]
                .iloc[-quarter:]
                .mean()
            )

            first_ratio = (
                first_mean / normal_first_mean
            )

            last_ratio = (
                last_mean / normal_last_mean
            )

            if (
                last_ratio - first_ratio
            ) < 0.18:
                raise ValueError(
                    "La dérive énergétique n'est pas "
                    "suffisamment progressive."
                )

            validation_results.append(
                f"{anomaly_type}: facteur final "
                f"{last_ratio:.2f}"
            )

    print("Validation des anomalies réussie.")
    print()
    print(
        f"Nombre d'événements : {len(labels)}"
    )
    print(
        f"Nombre de lignes étiquetées : "
        f"{len(truth):,}"
    )
    print(
        f"Nombre de lignes réellement modifiées : "
        f"{len(changed_ids):,}"
    )
    print(
        f"Couverture des étiquettes : "
        f"{changed_coverage:.2%}"
    )
    print()
    print("Contrôles par anomalie :")

    for result in validation_results:
        print(f"- {result}")


if __name__ == "__main__":
    main()