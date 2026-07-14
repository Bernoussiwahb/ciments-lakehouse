from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "industrial_measurements_normal.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "normal_data_analysis"
)


def load_data() -> pd.DataFrame:
    """Charge et prépare les données normales."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Exécute d'abord simulation/generate_normal_data.py."
        )

    dataframe = pd.read_parquet(DATA_PATH)

    if dataframe.empty:
        raise ValueError("Le fichier de données est vide.")

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )

    return dataframe.sort_values(
        ["equipment_id", "timestamp"]
    ).reset_index(drop=True)


def save_descriptive_statistics(
    dataframe: pd.DataFrame,
) -> None:
    """Enregistre les statistiques descriptives par équipement."""

    numeric_columns = [
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
        "rotation_speed_rpm",
        "alarm_count",
    ]

    statistics = (
        dataframe
        .groupby("equipment_id")[numeric_columns]
        .describe()
        .round(3)
    )

    statistics.to_csv(
        OUTPUT_DIRECTORY
        / "descriptive_statistics_by_equipment.csv",
        encoding="utf-8",
    )


def save_correlation_matrices(
    dataframe: pd.DataFrame,
) -> None:
    """Enregistre une matrice de corrélation pour chaque équipement."""

    correlation_columns = [
        "production_rate_tph",
        "electrical_power_kw",
        "specific_energy_kwh_t",
        "temperature_c",
        "pressure_bar",
        "vibration_mm_s",
        "current_ampere",
        "rotation_speed_rpm",
        "alarm_count",
    ]

    for equipment_id, subset in dataframe.groupby(
        "equipment_id"
    ):
        correlation = (
            subset[correlation_columns]
            .corr(numeric_only=True)
            .round(3)
        )

        correlation.to_csv(
            OUTPUT_DIRECTORY
            / f"correlation_{equipment_id}.csv",
            encoding="utf-8",
        )


def save_daily_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule et enregistre un résumé quotidien."""

    working = dataframe.copy()
    working["date"] = working["timestamp"].dt.date

    daily_summary = (
        working
        .groupby(["date", "equipment_id"])
        .agg(
            total_production_tonnes=(
                "production_tonnes",
                "sum",
            ),
            total_energy_kwh=(
                "electrical_energy_kwh",
                "sum",
            ),
            average_power_kw=(
                "electrical_power_kw",
                "mean",
            ),
            average_temperature_c=(
                "temperature_c",
                "mean",
            ),
            average_vibration_mm_s=(
                "vibration_mm_s",
                "mean",
            ),
            maintenance_rows=(
                "operating_state",
                lambda values: (
                    values == "MAINTENANCE"
                ).sum(),
            ),
        )
        .reset_index()
    )

    daily_summary["daily_specific_energy_kwh_t"] = (
        daily_summary["total_energy_kwh"]
        / daily_summary["total_production_tonnes"]
    )

    daily_summary.to_csv(
        OUTPUT_DIRECTORY / "daily_summary.csv",
        index=False,
        encoding="utf-8",
    )

    return daily_summary


def plot_weekly_timeseries(
    dataframe: pd.DataFrame,
) -> None:
    """Crée des séries temporelles sur une semaine pour chaque équipement."""

    first_timestamp = dataframe["timestamp"].min()
    end_timestamp = first_timestamp + pd.Timedelta(days=7)

    sample = dataframe[
        (dataframe["timestamp"] >= first_timestamp)
        & (dataframe["timestamp"] < end_timestamp)
    ]

    for equipment_id, subset in sample.groupby(
        "equipment_id"
    ):
        figure = plt.figure(figsize=(13, 5))
        axis = figure.add_subplot(111)

        axis.plot(
            subset["timestamp"],
            subset["production_rate_tph"],
            label="Production (t/h)",
        )

        axis.set_title(
            f"Production sur une semaine - {equipment_id}"
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("Production (t/h)")
        axis.grid(True)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()

        figure.savefig(
            OUTPUT_DIRECTORY
            / f"weekly_production_{equipment_id}.png",
            dpi=150,
        )
        plt.close(figure)

        figure = plt.figure(figsize=(13, 5))
        axis = figure.add_subplot(111)

        axis.plot(
            subset["timestamp"],
            subset["electrical_power_kw"],
            label="Puissance électrique (kW)",
        )

        axis.set_title(
            f"Puissance électrique sur une semaine - "
            f"{equipment_id}"
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("Puissance (kW)")
        axis.grid(True)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()

        figure.savefig(
            OUTPUT_DIRECTORY
            / f"weekly_power_{equipment_id}.png",
            dpi=150,
        )
        plt.close(figure)


def plot_production_energy_relationship(
    dataframe: pd.DataFrame,
) -> None:
    """Visualise la relation entre production et puissance."""

    for equipment_id, subset in dataframe.groupby(
        "equipment_id"
    ):
        running = subset[
            subset["operating_state"] == "RUNNING"
        ]

        # Échantillon limité pour garder un graphique lisible.
        if len(running) > 10000:
            running = running.sample(
                n=10000,
                random_state=42,
            )

        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111)

        axis.scatter(
            running["production_rate_tph"],
            running["electrical_power_kw"],
            alpha=0.25,
            s=10,
        )

        correlation = running[
            [
                "production_rate_tph",
                "electrical_power_kw",
            ]
        ].corr().iloc[0, 1]

        axis.set_title(
            f"Production et puissance - {equipment_id}\n"
            f"Corrélation = {correlation:.3f}"
        )
        axis.set_xlabel("Production (t/h)")
        axis.set_ylabel("Puissance électrique (kW)")
        axis.grid(True)
        figure.tight_layout()

        figure.savefig(
            OUTPUT_DIRECTORY
            / f"production_power_relationship_{equipment_id}.png",
            dpi=150,
        )
        plt.close(figure)


def plot_sensor_distributions(
    dataframe: pd.DataFrame,
) -> None:
    """Crée des histogrammes de température et vibration."""

    for equipment_id, subset in dataframe.groupby(
        "equipment_id"
    ):
        running = subset[
            subset["operating_state"] == "RUNNING"
        ]

        figure = plt.figure(figsize=(9, 6))
        axis = figure.add_subplot(111)

        axis.hist(
            running["temperature_c"],
            bins=40,
        )

        axis.set_title(
            f"Distribution de la température - {equipment_id}"
        )
        axis.set_xlabel("Température (°C)")
        axis.set_ylabel("Nombre de mesures")
        axis.grid(True)
        figure.tight_layout()

        figure.savefig(
            OUTPUT_DIRECTORY
            / f"temperature_distribution_{equipment_id}.png",
            dpi=150,
        )
        plt.close(figure)

        figure = plt.figure(figsize=(9, 6))
        axis = figure.add_subplot(111)

        axis.hist(
            running["vibration_mm_s"],
            bins=40,
        )

        axis.set_title(
            f"Distribution des vibrations - {equipment_id}"
        )
        axis.set_xlabel("Vibration (mm/s)")
        axis.set_ylabel("Nombre de mesures")
        axis.grid(True)
        figure.tight_layout()

        figure.savefig(
            OUTPUT_DIRECTORY
            / f"vibration_distribution_{equipment_id}.png",
            dpi=150,
        )
        plt.close(figure)


def plot_maintenance_example(
    dataframe: pd.DataFrame,
) -> None:
    """Montre l'effet d'une maintenance sur production et énergie."""

    maintenance_rows = dataframe[
        dataframe["operating_state"] == "MAINTENANCE"
    ]

    if maintenance_rows.empty:
        raise ValueError(
            "Aucune ligne de maintenance n'a été trouvée."
        )

    first_maintenance = maintenance_rows.iloc[0]
    equipment_id = first_maintenance["equipment_id"]
    maintenance_timestamp = first_maintenance["timestamp"]

    start_window = (
        maintenance_timestamp
        - pd.Timedelta(hours=6)
    )
    end_window = (
        maintenance_timestamp
        + pd.Timedelta(hours=8)
    )

    subset = dataframe[
        (dataframe["equipment_id"] == equipment_id)
        & (dataframe["timestamp"] >= start_window)
        & (dataframe["timestamp"] <= end_window)
    ]

    figure = plt.figure(figsize=(13, 5))
    axis = figure.add_subplot(111)

    axis.plot(
        subset["timestamp"],
        subset["production_rate_tph"],
        label="Production (t/h)",
    )
    axis.plot(
        subset["timestamp"],
        subset["electrical_power_kw"] / 30,
        label="Puissance divisée par 30",
    )

    axis.set_title(
        f"Effet d'une maintenance planifiée - {equipment_id}"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Valeur")
    axis.grid(True)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY / "planned_maintenance_example.png",
        dpi=150,
    )
    plt.close(figure)


def create_quality_report(
    dataframe: pd.DataFrame,
) -> None:
    """Crée un petit rapport de cohérence lisible."""

    report_lines: list[str] = []

    report_lines.append(
        "RAPPORT DE VALIDATION DES DONNÉES NORMALES"
    )
    report_lines.append("=" * 45)
    report_lines.append(
        f"Nombre total de lignes : {len(dataframe):,}"
    )
    report_lines.append(
        "Nombre d'équipements : "
        f"{dataframe['equipment_id'].nunique()}"
    )
    report_lines.append(
        "Date de début : "
        f"{dataframe['timestamp'].min()}"
    )
    report_lines.append(
        "Date de fin : "
        f"{dataframe['timestamp'].max()}"
    )
    total_missing_values = int(
        dataframe.isna().sum().sum()
    )
    expected_specific_energy_missing = int(
        (
            dataframe["specific_energy_kwh_t"].isna()
            & (dataframe["production_tonnes"] == 0)
        ).sum()
    )
    unexpected_missing_values = (
        total_missing_values
        - expected_specific_energy_missing
    )

    report_lines.append(
        "Valeurs manquantes totales : "
        f"{total_missing_values}"
    )
    report_lines.append(
        "Valeurs manquantes attendues de consommation spécifique "
        "(production nulle) : "
        f"{expected_specific_energy_missing}"
    )
    report_lines.append(
        "Valeurs manquantes inattendues : "
        f"{unexpected_missing_values}"
    )
    report_lines.append(
        "Doublons équipement/timestamp : "
        f"{int(dataframe.duplicated(['equipment_id', 'timestamp']).sum())}"
    )
    report_lines.append("")

    for equipment_id, subset in dataframe.groupby(
        "equipment_id"
    ):
        running = subset[
            subset["operating_state"] == "RUNNING"
        ]

        correlation = running[
            [
                "production_rate_tph",
                "electrical_power_kw",
            ]
        ].corr().iloc[0, 1]

        utilization = (
            running["capacity_utilization_pct"].mean()
        )

        maintenance_rows = int(
            (
                subset["operating_state"]
                == "MAINTENANCE"
            ).sum()
        )

        report_lines.append(
            f"Équipement : {equipment_id}"
        )
        report_lines.append(
            f"- Utilisation moyenne en marche : "
            f"{utilization:.2f} %"
        )
        report_lines.append(
            f"- Corrélation production/puissance : "
            f"{correlation:.3f}"
        )
        report_lines.append(
            f"- Lignes de maintenance : "
            f"{maintenance_rows}"
        )
        report_lines.append(
            f"- Température moyenne en marche : "
            f"{running['temperature_c'].mean():.2f} °C"
        )
        report_lines.append(
            f"- Vibration moyenne en marche : "
            f"{running['vibration_mm_s'].mean():.2f} mm/s"
        )
        report_lines.append("")

    report_path = (
        OUTPUT_DIRECTORY
        / "normal_data_validation_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8-sig",
    )


def main() -> None:
    """Lance toute l'analyse des données normales."""

    print("Chargement des données normales...")

    dataframe = load_data()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Création des statistiques descriptives...")
    save_descriptive_statistics(dataframe)

    print("Création des matrices de corrélation...")
    save_correlation_matrices(dataframe)

    print("Création du résumé quotidien...")
    save_daily_summary(dataframe)

    print("Création des graphiques hebdomadaires...")
    plot_weekly_timeseries(dataframe)

    print("Analyse de la relation production/énergie...")
    plot_production_energy_relationship(dataframe)

    print("Création des distributions des capteurs...")
    plot_sensor_distributions(dataframe)

    print("Création de l'exemple de maintenance...")
    plot_maintenance_example(dataframe)

    print("Création du rapport de validation...")
    create_quality_report(dataframe)

    print()
    print("Analyse terminée avec succès.")
    print(f"Résultats enregistrés dans : {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()