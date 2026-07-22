from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_feature_table"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_feature_columns.json"
)

ANOMALY_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "anomaly_labels.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest"
)

PREDICTION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_predictions"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "isolation_forest"
)

MODEL_MANIFEST_PATH = (
    MODEL_DIRECTORY
    / "model_manifest.json"
)

MODEL_VERSION = "isolation_forest_v1"
RANDOM_STATE = 42


def parse_arguments() -> argparse.Namespace:
    """Lit les paramètres d'entraînement."""

    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un modèle Isolation Forest "
            "local pour chaque équipement."
        )
    )

    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.01,
        help=(
            "Quantile des scores normaux utilisé comme "
            "seuil d'anomalie. Valeur par défaut : 0.01."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help=(
            "Nombre d'arbres Isolation Forest. "
            "Valeur par défaut : 300."
        ),
    )

    arguments = parser.parse_args()

    if not 0 < arguments.threshold_quantile < 0.5:
        raise ValueError(
            "threshold-quantile doit être compris "
            "strictement entre 0 et 0.5."
        )

    if arguments.n_estimators < 50:
        raise ValueError(
            "n-estimators doit être supérieur ou égal à 50."
        )

    return arguments


def read_partitioned_dataset(
    directory: Path,
) -> pd.DataFrame:
    """Charge la table de variables IA."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    parquet_files = list(
        directory.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"Aucun fichier Parquet trouvé dans : {directory}"
        )

    dataset = ds.dataset(
        str(directory),
        format="parquet",
        partitioning="hive",
    )

    dataframe = (
        dataset
        .to_table()
        .to_pandas()
    )

    if dataframe.empty:
        raise ValueError(
            "La table de variables IA est vide."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )

    if dataframe["measurement_id"].duplicated().any():
        raise ValueError(
            "measurement_id contient des doublons."
        )

    return dataframe.sort_values(
        [
            "equipment_id",
            "timestamp",
        ]
    ).reset_index(drop=True)


def load_feature_columns() -> list[str]:
    """Charge la liste officielle des variables du modèle."""

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            "ml_feature_columns.json est introuvable. "
            "Exécute d'abord build_ml_features.py."
        )

    payload = json.loads(
        FEATURE_LIST_PATH.read_text(
            encoding="utf-8"
        )
    )

    features = payload.get(
        "feature_columns"
    )

    if not features:
        raise ValueError(
            "La liste des variables du modèle est vide."
        )

    forbidden_columns = {
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "severity",
        "dataset_split",
        "is_training_candidate",
    }

    leakage_columns = (
        set(features)
        & forbidden_columns
    )

    if leakage_columns:
        raise ValueError(
            "Fuite de données : colonnes interdites "
            f"dans les variables IA : {sorted(leakage_columns)}"
        )

    return list(features)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """Effectue une division sûre."""

    if denominator == 0:
        return None

    return numerator / denominator


def compute_binary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    anomaly_score: np.ndarray,
) -> dict[str, Any]:
    """Calcule des mesures binaires sans masquer les cas non définis."""

    truth = truth.astype(int)
    prediction = prediction.astype(int)

    tp = int(
        ((truth == 1) & (prediction == 1)).sum()
    )

    tn = int(
        ((truth == 0) & (prediction == 0)).sum()
    )

    fp = int(
        ((truth == 0) & (prediction == 1)).sum()
    )

    fn = int(
        ((truth == 1) & (prediction == 0)).sum()
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    if (
        precision is None
        or recall is None
        or precision + recall == 0
    ):
        f1_score = None
    else:
        f1_score = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    unique_truth = np.unique(truth)

    if len(unique_truth) == 2:
        roc_auc = float(
            roc_auc_score(
                truth,
                anomaly_score,
            )
        )

        average_precision = float(
            average_precision_score(
                truth,
                anomaly_score,
            )
        )
    else:
        roc_auc = None
        average_precision = None

    return {
        "evaluated_row_count": int(
            len(truth)
        ),
        "actual_anomaly_count": int(
            truth.sum()
        ),
        "predicted_anomaly_count": int(
            prediction.sum()
        ),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "false_positive_rate": (
            false_positive_rate
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def create_model_bundle(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    equipment_id: str,
    threshold_quantile: float,
    n_estimators: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Entraîne le modèle d'un équipement."""

    feature_matrix = (
        training_data[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if feature_matrix.isna().any().any():
        columns = feature_matrix.columns[
            feature_matrix.isna().any()
        ].tolist()

        raise ValueError(
            f"Valeurs manquantes dans le train "
            f"de {equipment_id} : {columns}"
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = RobustScaler()

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    imputed_matrix = imputer.fit_transform(
        feature_matrix
    )

    scaled_matrix = scaler.fit_transform(
        imputed_matrix
    )

    model.fit(
        scaled_matrix
    )

    training_scores = model.score_samples(
        scaled_matrix
    )

    decision_threshold = float(
        np.quantile(
            training_scores,
            threshold_quantile,
        )
    )

    predicted_training_anomalies = (
        training_scores
        <= decision_threshold
    )

    bundle = {
        "model_version": MODEL_VERSION,
        "equipment_id": equipment_id,
        "feature_columns": feature_columns,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "decision_threshold": (
            decision_threshold
        ),
        "threshold_quantile": (
            threshold_quantile
        ),
        "random_state": RANDOM_STATE,
        "n_estimators": n_estimators,
    }

    metadata = {
        "equipment_id": equipment_id,
        "training_row_count": int(
            len(training_data)
        ),
        "training_minimum_timestamp": (
            training_data[
                "timestamp"
            ].min().isoformat()
        ),
        "training_maximum_timestamp": (
            training_data[
                "timestamp"
            ].max().isoformat()
        ),
        "decision_threshold": (
            decision_threshold
        ),
        "training_score_minimum": float(
            training_scores.min()
        ),
        "training_score_mean": float(
            training_scores.mean()
        ),
        "training_score_maximum": float(
            training_scores.max()
        ),
        "training_predicted_anomaly_count": int(
            predicted_training_anomalies.sum()
        ),
        "training_predicted_anomaly_rate_pct": float(
            predicted_training_anomalies.mean()
            * 100
        ),
    }

    return bundle, metadata


def score_equipment(
    equipment_data: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Calcule les scores d'un équipement."""

    result = equipment_data[
        [
            "measurement_id",
            "timestamp",
            "equipment_id",
            "operating_state",
            "dataset_split",
            "is_training_candidate",
            "is_anomaly",
            "anomaly_id",
            "anomaly_type",
            "severity",
        ]
    ].copy()

    result = result.rename(
        columns={
            "is_anomaly": (
                "ground_truth_is_anomaly"
            ),
            "anomaly_id": (
                "ground_truth_anomaly_id"
            ),
            "anomaly_type": (
                "ground_truth_anomaly_type"
            ),
            "severity": (
                "ground_truth_severity"
            ),
        }
    )

    result["is_scored"] = (
        equipment_data[
            "operating_state"
        ].eq("RUNNING")
    )

    result["iforest_score"] = np.nan
    result["anomaly_margin"] = np.nan
    result["decision_threshold"] = float(
        bundle["decision_threshold"]
    )

    result["predicted_is_anomaly"] = (
        pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int8",
        )
    )

    scored_mask = result[
        "is_scored"
    ].astype(bool)

    if scored_mask.any():
        matrix = (
            equipment_data.loc[
                scored_mask,
                bundle["feature_columns"],
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        imputed = bundle[
            "imputer"
        ].transform(matrix)

        scaled = bundle[
            "scaler"
        ].transform(imputed)

        scores = bundle[
            "model"
        ].score_samples(scaled)

        threshold = float(
            bundle["decision_threshold"]
        )

        predictions = (
            scores <= threshold
        ).astype("int8")

        result.loc[
            scored_mask,
            "iforest_score",
        ] = scores

        result.loc[
            scored_mask,
            "anomaly_margin",
        ] = threshold - scores

        result.loc[
            scored_mask,
            "predicted_is_anomaly",
        ] = predictions

    result["model_version"] = (
        MODEL_VERSION
    )

    result["evaluation_scope"] = (
        result["dataset_split"].eq("test")
        & result["is_scored"]
    )

    return result


def prepare_output_directories() -> None:
    """Réinitialise les artefacts du modèle."""

    for directory in [
        MODEL_DIRECTORY,
        PREDICTION_DIRECTORY,
        REPORT_DIRECTORY,
    ]:
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def write_predictions(
    predictions: pd.DataFrame,
) -> int:
    """Écrit les prédictions en Parquet partitionné."""

    output = predictions.copy()

    output["prediction_year"] = (
        output["timestamp"]
        .dt.year
        .astype("int16")
    )

    output["prediction_month"] = (
        output["timestamp"]
        .dt.month
        .astype("int8")
    )

    table = pa.Table.from_pandas(
        output,
        preserve_index=False,
    )

    ds.write_dataset(
        data=table,
        base_dir=str(
            PREDICTION_DIRECTORY
        ),
        format="parquet",
        partitioning=[
            "dataset_split",
            "equipment_id",
            "prediction_year",
            "prediction_month",
        ],
        partitioning_flavor="hive",
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=50_000,
        max_rows_per_group=50_000,
    )

    return len(
        list(
            PREDICTION_DIRECTORY.rglob(
                "*.parquet"
            )
        )
    )


def build_event_detection_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule la détection des événements présents dans le test."""

    columns = [
        "anomaly_id",
        "equipment_id",
        "anomaly_type",
        "severity",
        "start_time",
        "end_time",
        "event_row_count",
        "scored_row_count",
        "predicted_anomaly_row_count",
        "detected",
        "detection_coverage_pct",
        "minimum_anomaly_margin",
        "maximum_anomaly_margin",
    ]

    if not ANOMALY_LABELS_PATH.exists():
        return pd.DataFrame(
            columns=columns
        )

    labels = pd.read_csv(
        ANOMALY_LABELS_PATH,
        parse_dates=[
            "start_time",
            "end_time",
        ],
    )

    rows: list[dict[str, Any]] = []

    for event in labels.itertuples(
        index=False
    ):
        mask = (
            predictions[
                "equipment_id"
            ].eq(event.equipment_id)
            & predictions[
                "timestamp"
            ].between(
                event.start_time,
                event.end_time,
                inclusive="both",
            )
            & predictions[
                "dataset_split"
            ].eq("test")
        )

        event_predictions = (
            predictions.loc[mask]
        )

        if event_predictions.empty:
            continue

        scored = event_predictions[
            event_predictions[
                "is_scored"
            ]
        ]

        predicted_count = int(
            scored[
                "predicted_is_anomaly"
            ]
            .fillna(0)
            .astype(int)
            .sum()
        )

        event_row_count = int(
            len(event_predictions)
        )

        scored_row_count = int(
            len(scored)
        )

        coverage = (
            predicted_count
            / event_row_count
            * 100
            if event_row_count
            else 0.0
        )

        rows.append(
            {
                "anomaly_id": event.anomaly_id,
                "equipment_id": event.equipment_id,
                "anomaly_type": event.anomaly_type,
                "severity": event.severity,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "event_row_count": event_row_count,
                "scored_row_count": scored_row_count,
                "predicted_anomaly_row_count": (
                    predicted_count
                ),
                "detected": (
                    predicted_count > 0
                ),
                "detection_coverage_pct": round(
                    coverage,
                    3,
                ),
                "minimum_anomaly_margin": (
                    float(
                        scored[
                            "anomaly_margin"
                        ].min()
                    )
                    if not scored.empty
                    else np.nan
                ),
                "maximum_anomaly_margin": (
                    float(
                        scored[
                            "anomaly_margin"
                        ].max()
                    )
                    if not scored.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def build_metrics_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les métriques globales et par équipement."""

    evaluation = predictions[
        predictions[
            "evaluation_scope"
        ]
    ].copy()

    if evaluation.empty:
        raise ValueError(
            "Aucune ligne disponible pour l'évaluation."
        )

    if evaluation[
        "predicted_is_anomaly"
    ].isna().any():
        raise ValueError(
            "Certaines lignes d'évaluation ne "
            "possèdent pas de prédiction."
        )

    rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, pd.DataFrame]] = [
        ("OVERALL", evaluation)
    ]

    scopes.extend(
        [
            (
                str(equipment_id),
                subset,
            )
            for (
                equipment_id,
                subset,
            ) in evaluation.groupby(
                "equipment_id",
                sort=True,
            )
        ]
    )

    for scope_name, subset in scopes:
        metrics = compute_binary_metrics(
            truth=subset[
                "ground_truth_is_anomaly"
            ].to_numpy(dtype=int),
            prediction=subset[
                "predicted_is_anomaly"
            ].to_numpy(dtype=int),
            anomaly_score=subset[
                "anomaly_margin"
            ].to_numpy(dtype=float),
        )

        rows.append(
            {
                "scope": scope_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def write_text_report(
    metrics: pd.DataFrame,
    event_summary: pd.DataFrame,
    model_metadata: list[dict[str, Any]],
    threshold_quantile: float,
    n_estimators: int,
) -> Path:
    """Écrit un rapport lisible des performances."""

    overall = metrics[
        metrics["scope"].eq("OVERALL")
    ].iloc[0]

    detected_event_count = int(
        event_summary[
            "detected"
        ].sum()
        if not event_summary.empty
        else 0
    )

    total_event_count = int(
        len(event_summary)
    )

    def display_metric(
        value: Any,
    ) -> str:
        if pd.isna(value):
            return "Non défini"

        return f"{float(value):.4f}"

    lines = [
        "RAPPORT D'ENTRAÎNEMENT ET D'ÉVALUATION ISOLATION FOREST",
        "=" * 61,
        f"Version du modèle : {MODEL_VERSION}",
        f"Nombre de modèles : {len(model_metadata)}",
        f"Arbres par modèle : {n_estimators}",
        (
            "Quantile du seuil normal : "
            f"{threshold_quantile:.4f}"
        ),
        (
            "Périmètre d'évaluation : "
            "lignes RUNNING du jeu de test"
        ),
        "",
        "RÉSULTATS GLOBAUX",
        "-" * 20,
        (
            "Lignes évaluées : "
            f"{int(overall['evaluated_row_count']):,}"
        ),
        (
            "Anomalies réelles : "
            f"{int(overall['actual_anomaly_count']):,}"
        ),
        (
            "Anomalies prédites : "
            f"{int(overall['predicted_anomaly_count']):,}"
        ),
        (
            "Vrais positifs : "
            f"{int(overall['true_positive']):,}"
        ),
        (
            "Faux positifs : "
            f"{int(overall['false_positive']):,}"
        ),
        (
            "Faux négatifs : "
            f"{int(overall['false_negative']):,}"
        ),
        (
            "Précision : "
            f"{display_metric(overall['precision'])}"
        ),
        (
            "Rappel : "
            f"{display_metric(overall['recall'])}"
        ),
        (
            "F1-score : "
            f"{display_metric(overall['f1_score'])}"
        ),
        (
            "Spécificité : "
            f"{display_metric(overall['specificity'])}"
        ),
        (
            "ROC-AUC : "
            f"{display_metric(overall['roc_auc'])}"
        ),
        (
            "Average Precision : "
            f"{display_metric(overall['average_precision'])}"
        ),
        "",
        "DÉTECTION DES ÉVÉNEMENTS DU JEU DE TEST",
        "-" * 39,
        (
            "Événements détectés : "
            f"{detected_event_count}/{total_event_count}"
        ),
        "",
        "LIMITES",
        "-" * 8,
        (
            "- Les données et anomalies utilisées sont synthétiques."
        ),
        (
            "- Les étiquettes servent uniquement à créer un "
            "benchmark propre et à évaluer le modèle."
        ),
        (
            "- Ces résultats ne prouvent pas une performance "
            "sur des données industrielles réelles."
        ),
    ]

    report_path = (
        REPORT_DIRECTORY
        / "isolation_forest_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

    return report_path


def main() -> None:
    """Entraîne, prédit et évalue les modèles."""

    arguments = parse_arguments()

    print(
        "Chargement de la table de variables IA..."
    )

    features = read_partitioned_dataset(
        FEATURE_DATASET_DIRECTORY
    )

    feature_columns = load_feature_columns()

    missing_feature_columns = (
        set(feature_columns)
        - set(features.columns)
    )

    if missing_feature_columns:
        raise ValueError(
            "Variables manquantes dans la table IA : "
            f"{sorted(missing_feature_columns)}"
        )

    prepare_output_directories()

    predictions_list: list[
        pd.DataFrame
    ] = []

    model_metadata: list[
        dict[str, Any]
    ] = []

    equipment_ids = sorted(
        features[
            "equipment_id"
        ].astype(str).unique()
    )

    print(
        f"Entraînement de {len(equipment_ids)} "
        "modèles locaux..."
    )

    for equipment_id in equipment_ids:
        print(
            f"- Modèle pour {equipment_id}..."
        )

        equipment_data = features[
            features[
                "equipment_id"
            ].astype(str).eq(equipment_id)
        ].copy()

        training_data = equipment_data[
            equipment_data[
                "is_training_candidate"
            ].astype(bool)
        ].copy()

        if training_data.empty:
            raise ValueError(
                "Aucune donnée d'entraînement pour "
                f"{equipment_id}."
            )

        bundle, metadata = create_model_bundle(
            training_data=training_data,
            feature_columns=feature_columns,
            equipment_id=equipment_id,
            threshold_quantile=(
                arguments.threshold_quantile
            ),
            n_estimators=(
                arguments.n_estimators
            ),
        )

        model_path = (
            MODEL_DIRECTORY
            / f"{equipment_id}.joblib"
        )

        joblib.dump(
            bundle,
            model_path,
            compress=3,
        )

        metadata["model_path"] = str(
            model_path
        )

        model_metadata.append(
            metadata
        )

        equipment_predictions = (
            score_equipment(
                equipment_data=equipment_data,
                bundle=bundle,
            )
        )

        predictions_list.append(
            equipment_predictions
        )

    predictions = (
        pd.concat(
            predictions_list,
            ignore_index=True,
        )
        .sort_values(
            [
                "equipment_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    metrics = build_metrics_table(
        predictions
    )

    event_summary = (
        build_event_detection_summary(
            predictions
        )
    )

    prediction_file_count = (
        write_predictions(
            predictions
        )
    )

    metrics.to_csv(
        REPORT_DIRECTORY
        / "isolation_forest_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_summary.to_csv(
        REPORT_DIRECTORY
        / "event_detection_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    pd.DataFrame(
        model_metadata
    ).to_csv(
        REPORT_DIRECTORY
        / "model_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    processed_at_utc = datetime.now(
        timezone.utc
    )

    feature_source_batch_ids = (
        features[
            "feature_processing_batch_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if (
            "feature_processing_batch_id"
            in features.columns
        )
        else []
    )

    manifest_seed = (
        MODEL_VERSION
        + processed_at_utc.isoformat()
    )

    model_batch_id = (
        "MODEL_"
        + hashlib.sha256(
            manifest_seed.encode("utf-8")
        ).hexdigest()[:16].upper()
    )

    manifest = {
        "model_batch_id": model_batch_id,
        "model_version": MODEL_VERSION,
        "status": "SUCCESS",
        "trained_at_utc": (
            processed_at_utc.isoformat()
        ),
        "algorithm": "IsolationForest",
        "strategy": "one_model_per_equipment",
        "random_state": RANDOM_STATE,
        "n_estimators": (
            arguments.n_estimators
        ),
        "threshold_quantile": (
            arguments.threshold_quantile
        ),
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": (
            feature_columns
        ),
        "source_feature_batch_ids": (
            feature_source_batch_ids
        ),
        "model_count": len(
            model_metadata
        ),
        "models": model_metadata,
        "prediction_dataset_directory": str(
            PREDICTION_DIRECTORY
        ),
        "prediction_file_count": int(
            prediction_file_count
        ),
        "metrics_path": str(
            REPORT_DIRECTORY
            / "isolation_forest_metrics.csv"
        ),
        "event_summary_path": str(
            REPORT_DIRECTORY
            / "event_detection_summary.csv"
        ),
        "evaluation_policy": {
            "scope": (
                "dataset_split=test AND "
                "operating_state=RUNNING"
            ),
            "labels_used_for_training_features": False,
            "labels_used_to_exclude_synthetic_anomalies_from_train": True,
            "labels_used_for_evaluation": True,
        },
        "limitations": [
            (
                "Synthetic industrial data and synthetic "
                "anomalies only."
            ),
            (
                "Performance cannot be generalized to "
                "real Ciments du Maroc data."
            ),
        ],
    }

    MODEL_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = write_text_report(
        metrics=metrics,
        event_summary=event_summary,
        model_metadata=model_metadata,
        threshold_quantile=(
            arguments.threshold_quantile
        ),
        n_estimators=(
            arguments.n_estimators
        ),
    )

    overall = metrics[
        metrics["scope"].eq("OVERALL")
    ].iloc[0]

    print()
    print(
        "Entraînement Isolation Forest réussi."
    )
    print(
        f"Modèles créés : "
        f"{len(model_metadata)}"
    )
    print(
        f"Lignes évaluées : "
        f"{int(overall['evaluated_row_count']):,}"
    )
    print(
        f"Précision : "
        f"{overall['precision']:.4f}"
        if pd.notna(
            overall["precision"]
        )
        else "Précision : non définie"
    )
    print(
        f"Rappel : "
        f"{overall['recall']:.4f}"
        if pd.notna(
            overall["recall"]
        )
        else "Rappel : non défini"
    )
    print(
        f"F1-score : "
        f"{overall['f1_score']:.4f}"
        if pd.notna(
            overall["f1_score"]
        )
        else "F1-score : non défini"
    )
    print(
        f"Rapport : {report_path}"
    )
    print(
        f"Modèles : {MODEL_DIRECTORY}"
    )
    print(
        f"Prédictions : {PREDICTION_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
