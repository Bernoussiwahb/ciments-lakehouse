from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "anomaly_visualization"
)


ANOMALY_VARIABLES = {
    "ENERGY_OVERCONSUMPTION": {
        "column": "electrical_power_kw",
        "label": "Puissance électrique (kW)",
    },
    "PRODUCTION_DROP": {
        "column": "production_rate_tph",
        "label": "Production (t/h)",
    },
    "OVERHEATING": {
        "column": "temperature_c",
        "label": "Température (°C)",
    },
    "EXCESSIVE_VIBRATION": {
        "column": "vibration_mm_s",
        "label": "Vibration (mm/s)",
    },
    "PRESSURE_DROP": {
        "column": "pressure_bar",
        "label": "Pression (bar)",
    },
    "SENSOR_STUCK": {
        "column": "temperature_c",
        "label": "Température (°C)",
    },
    "ENERGY_DRIFT": {
        "column": "electrical_power_kw",
        "label": "Puissance électrique (kW)",
    },
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les jeux de données normal, anormal et les événements."""

    missing_paths = [
        path
        for path in [
            NORMAL_PATH,
            ANOMALOUS_PATH,
            LABELS_PATH,
        ]
        if not path.exists()
    ]

    if missing_paths:
        formatted_paths = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Fichiers manquants :\n"
            f"{formatted_paths}\n"
            "Exécute d'abord generate_normal_data.py "
            "puis inject_anomalies.py."
        )

    normal = pd.read_parquet(NORMAL_PATH)
    anomalous = pd.read_parquet(ANOMALOUS_PATH)

    labels = pd.read_csv(
        LABELS_PATH,
        parse_dates=[
            "start_time",
            "end_time",
        ],
    )

    normal["timestamp"] = pd.to_datetime(
        normal["timestamp"],
        errors="raise",
    )

    anomalous["timestamp"] = pd.to_datetime(
        anomalous["timestamp"],
        errors="raise",
    )

    if normal["measurement_id"].duplicated().any():
        raise ValueError(
            "Le jeu de données normal contient "
            "des measurement_id dupliqués."
        )

    if anomalous["measurement_id"].duplicated().any():
        raise ValueError(
            "Le jeu de données anormal contient "
            "des measurement_id dupliqués."
        )

    normal_ids = set(normal["measurement_id"])
    anomalous_ids = set(anomalous["measurement_id"])

    if normal_ids != anomalous_ids:
        raise ValueError(
            "Les deux jeux de données ne contiennent "
            "pas les mêmes measurement_id."
        )

    anomalous = (
        anomalous
        .set_index("measurement_id")
        .loc[normal["measurement_id"]]
        .reset_index()
    )

    return normal, anomalous, labels


def build_analysis_window(
    event: pd.Series,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Ajoute une marge temporelle avant et après l'anomalie
    afin de visualiser le fonctionnement normal autour.
    """

    duration = (
        event["end_time"]
        - event["start_time"]
    )

    margin = max(
        pd.Timedelta(hours=4),
        duration * 0.25,
    )

    return (
        event["start_time"] - margin,
        event["end_time"] + margin,
    )


def calculate_event_summary(
    event: pd.Series,
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
) -> dict:
    """Calcule un résumé numérique pour une anomalie."""

    anomaly_type = event["anomaly_type"]

    if anomaly_type not in ANOMALY_VARIABLES:
        raise KeyError(
            f"Type d'anomalie inconnu : {anomaly_type}"
        )

    column = ANOMALY_VARIABLES[
        anomaly_type
    ]["column"]

    event_mask = (
        (normal["equipment_id"] == event["equipment_id"])
        & (normal["timestamp"] >= event["start_time"])
        & (normal["timestamp"] <= event["end_time"])
    )

    normal_values = normal.loc[
        event_mask,
        column,
    ].astype(float)

    anomalous_values = anomalous.loc[
        event_mask,
        column,
    ].astype(float)

    if normal_values.empty:
        raise ValueError(
            f"Aucune donnée trouvée pour "
            f"{event['anomaly_id']}."
        )

    normal_mean = float(
        normal_values.mean()
    )

    anomalous_mean = float(
        anomalous_values.mean()
    )

    absolute_difference = (
        anomalous_mean - normal_mean
    )

    if np.isclose(normal_mean, 0):
        ratio = np.nan
        percentage_change = np.nan
    else:
        ratio = anomalous_mean / normal_mean
        percentage_change = (
            absolute_difference
            / normal_mean
            * 100
        )

    return {
        "anomaly_id": event["anomaly_id"],
        "equipment_id": event["equipment_id"],
        "anomaly_type": anomaly_type,
        "severity": event["severity"],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "affected_row_count": int(
            event["affected_row_count"]
        ),
        "analyzed_column": column,
        "normal_mean": round(
            normal_mean,
            3,
        ),
        "anomalous_mean": round(
            anomalous_mean,
            3,
        ),
        "absolute_difference": round(
            absolute_difference,
            3,
        ),
        "ratio": (
            round(float(ratio), 3)
            if not np.isnan(ratio)
            else np.nan
        ),
        "percentage_change": (
            round(
                float(percentage_change),
                2,
            )
            if not np.isnan(
                percentage_change
            )
            else np.nan
        ),
    }


def create_anomaly_plot(
    event: pd.Series,
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
) -> None:
    """Crée une comparaison visuelle pour une anomalie."""

    anomaly_type = event["anomaly_type"]

    if anomaly_type not in ANOMALY_VARIABLES:
        raise KeyError(
            f"Type d'anomalie inconnu : {anomaly_type}"
        )

    column = ANOMALY_VARIABLES[
        anomaly_type
    ]["column"]

    y_axis_label = ANOMALY_VARIABLES[
        anomaly_type
    ]["label"]

    window_start, window_end = (
        build_analysis_window(event)
    )

    mask = (
        (normal["equipment_id"] == event["equipment_id"])
        & (normal["timestamp"] >= window_start)
        & (normal["timestamp"] <= window_end)
    )

    normal_subset = normal.loc[
        mask,
        [
            "timestamp",
            column,
        ],
    ]

    anomalous_subset = anomalous.loc[
        mask,
        [
            "timestamp",
            column,
        ],
    ]

    if normal_subset.empty:
        raise ValueError(
            f"Aucune donnée disponible pour "
            f"{event['anomaly_id']}."
        )

    figure = plt.figure(
        figsize=(14, 6)
    )

    axis = figure.add_subplot(111)

    axis.plot(
        normal_subset["timestamp"],
        normal_subset[column],
        label="Données normales",
        linewidth=1.5,
    )

    axis.plot(
        anomalous_subset["timestamp"],
        anomalous_subset[column],
        label="Données avec anomalie",
        linewidth=1.5,
    )

    axis.axvspan(
        event["start_time"],
        event["end_time"],
        alpha=0.20,
        label="Période injectée",
    )

    axis.set_title(
        f"{event['anomaly_id']} — "
        f"{anomaly_type} — "
        f"{event['equipment_id']}"
    )

    axis.set_xlabel("Date et heure")
    axis.set_ylabel(y_axis_label)
    axis.grid(True)
    axis.legend()

    figure.autofmt_xdate()
    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / (
            f"{event['anomaly_id']}_"
            f"{anomaly_type}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def create_event_detail_files(
    event: pd.Series,
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
) -> None:
    """Crée un petit CSV comparatif pour chaque anomalie."""

    anomaly_type = event["anomaly_type"]
    column = ANOMALY_VARIABLES[
        anomaly_type
    ]["column"]

    window_start, window_end = (
        build_analysis_window(event)
    )

    mask = (
        (normal["equipment_id"] == event["equipment_id"])
        & (normal["timestamp"] >= window_start)
        & (normal["timestamp"] <= window_end)
    )

    details = pd.DataFrame(
        {
            "measurement_id": normal.loc[
                mask,
                "measurement_id",
            ].to_numpy(),
            "timestamp": normal.loc[
                mask,
                "timestamp",
            ].to_numpy(),
            "equipment_id": normal.loc[
                mask,
                "equipment_id",
            ].to_numpy(),
            "anomaly_id": event["anomaly_id"],
            "anomaly_type": anomaly_type,
            "is_in_anomaly_window": (
                (
                    normal.loc[
                        mask,
                        "timestamp",
                    ]
                    >= event["start_time"]
                )
                & (
                    normal.loc[
                        mask,
                        "timestamp",
                    ]
                    <= event["end_time"]
                )
            ).to_numpy(),
            "normal_value": normal.loc[
                mask,
                column,
            ].to_numpy(),
            "anomalous_value": anomalous.loc[
                mask,
                column,
            ].to_numpy(),
        }
    )

    details["difference"] = (
        details["anomalous_value"]
        - details["normal_value"]
    )

    details.to_csv(
        OUTPUT_DIRECTORY
        / (
            f"{event['anomaly_id']}_"
            f"{anomaly_type}_details.csv"
        ),
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )


def create_text_report(
    summary: pd.DataFrame,
) -> None:
    """Crée un rapport simple sur les anomalies."""

    lines: list[str] = []

    lines.append(
        "RAPPORT DE VISUALISATION DES ANOMALIES"
    )
    lines.append("=" * 45)
    lines.append(
        f"Nombre d'événements analysés : "
        f"{len(summary)}"
    )
    lines.append(
        f"Nombre total de lignes concernées : "
        f"{int(summary['affected_row_count'].sum()):,}"
    )
    lines.append("")

    for row in summary.itertuples(
        index=False
    ):
        lines.append(
            f"{row.anomaly_id} — "
            f"{row.anomaly_type}"
        )
        lines.append(
            f"- Équipement : {row.equipment_id}"
        )
        lines.append(
            f"- Gravité : {row.severity}"
        )
        lines.append(
            f"- Variable analysée : "
            f"{row.analyzed_column}"
        )
        lines.append(
            f"- Moyenne normale : "
            f"{row.normal_mean}"
        )
        lines.append(
            f"- Moyenne avec anomalie : "
            f"{row.anomalous_mean}"
        )
        lines.append(
            f"- Différence : "
            f"{row.absolute_difference}"
        )

        if not pd.isna(row.percentage_change):
            lines.append(
                f"- Variation : "
                f"{row.percentage_change:.2f} %"
            )

        lines.append("")

    report_path = (
        OUTPUT_DIRECTORY
        / "anomaly_visualization_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main() -> None:
    """Lance toute la visualisation des anomalies."""

    print(
        "Chargement des données normales "
        "et anormales..."
    )

    normal, anomalous, labels = load_data()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict] = []

    for _, event in labels.sort_values(
        "anomaly_id"
    ).iterrows():
        print(
            f"Visualisation de "
            f"{event['anomaly_id']} — "
            f"{event['anomaly_type']}..."
        )

        summary_rows.append(
            calculate_event_summary(
                event=event,
                normal=normal,
                anomalous=anomalous,
            )
        )

        create_anomaly_plot(
            event=event,
            normal=normal,
            anomalous=anomalous,
        )

        create_event_detail_files(
            event=event,
            normal=normal,
            anomalous=anomalous,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIRECTORY
        / "anomaly_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    create_text_report(
        summary
    )

    print()
    print(
        "Visualisation terminée avec succès."
    )
    print(
        f"Résultats enregistrés dans : "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()