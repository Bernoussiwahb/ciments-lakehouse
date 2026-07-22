from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


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

PREDICTION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "ml_predictions"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest"
)

MODEL_MANIFEST_PATH = (
    MODEL_DIRECTORY
    / "model_manifest.json"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "isolation_forest"
    / "isolation_forest_metrics.csv"
)

VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "isolation_forest"
    / "isolation_forest_validation_report.txt"
)


def read_dataset(
    directory: Path,
) -> pd.DataFrame:
    """Charge un dataset Parquet partitionné."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {directory}"
        )

    parquet_files = list(
        directory.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"Aucun fichier Parquet dans : {directory}"
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

    return dataframe


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Charge un document JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier JSON introuvable : {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """Effectue une division sûre."""

    if denominator == 0:
        return None

    return numerator / denominator


def recompute_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    anomaly_score: np.ndarray,
) -> dict[str, Any]:
    """Recalcule les métriques globales."""

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

    if len(np.unique(truth)) == 2:
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


def validate_feature_integrity(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Valide les clés et la couverture des prédictions."""

    if features["measurement_id"].duplicated().any():
        raise ValueError(
            "La table IA contient des identifiants dupliqués."
        )

    if predictions["measurement_id"].duplicated().any():
        raise ValueError(
            "Les prédictions contiennent des "
            "identifiants dupliqués."
        )

    if len(features) != len(predictions):
        raise ValueError(
            "Le nombre de prédictions diffère "
            "du nombre de lignes IA."
        )

    if set(
        features["measurement_id"]
    ) != set(
        predictions["measurement_id"]
    ):
        raise ValueError(
            "Les identifiants des prédictions "
            "ne correspondent pas à la table IA."
        )


def validate_model_files(
    manifest: dict[str, Any],
    feature_columns: list[str],
    equipment_ids: list[str],
) -> None:
    """Vérifie la présence et le contenu des modèles."""

    if int(
        manifest["model_count"]
    ) != len(equipment_ids):
        raise ValueError(
            "Nombre de modèles incorrect dans le manifeste."
        )

    manifest_features = list(
        manifest["feature_columns"]
    )

    if manifest_features != feature_columns:
        raise ValueError(
            "La liste de variables du manifeste "
            "ne correspond pas à la liste officielle."
        )

    forbidden_columns = {
        "is_anomaly",
        "anomaly_id",
        "anomaly_type",
        "severity",
    }

    if (
        set(manifest_features)
        & forbidden_columns
    ):
        raise ValueError(
            "Une étiquette d'évaluation est présente "
            "dans les variables du modèle."
        )

    for equipment_id in equipment_ids:
        model_path = (
            MODEL_DIRECTORY
            / f"{equipment_id}.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}"
            )

        bundle = joblib.load(
            model_path
        )

        required_keys = {
            "equipment_id",
            "feature_columns",
            "imputer",
            "scaler",
            "model",
            "decision_threshold",
            "threshold_quantile",
        }

        missing_keys = (
            required_keys
            - set(bundle)
        )

        if missing_keys:
            raise ValueError(
                f"Clés manquantes dans le modèle "
                f"{equipment_id} : {sorted(missing_keys)}"
            )

        if str(
            bundle["equipment_id"]
        ) != equipment_id:
            raise ValueError(
                "Le modèle ne correspond pas "
                f"à l'équipement {equipment_id}."
            )

        if list(
            bundle["feature_columns"]
        ) != feature_columns:
            raise ValueError(
                "Liste de variables incorrecte "
                f"dans le modèle {equipment_id}."
            )


def validate_prediction_rules(
    predictions: pd.DataFrame,
) -> None:
    """Vérifie les règles de calcul des prédictions."""

    running_mask = predictions[
        "operating_state"
    ].eq("RUNNING")

    if not predictions.loc[
        running_mask,
        "is_scored",
    ].all():
        raise ValueError(
            "Certaines lignes RUNNING ne sont pas scorées."
        )

    if predictions.loc[
        ~running_mask,
        "is_scored",
    ].any():
        raise ValueError(
            "Des lignes non RUNNING ont été scorées."
        )

    scored = predictions[
        predictions["is_scored"]
    ]

    if scored[
        "iforest_score"
    ].isna().any():
        raise ValueError(
            "Score manquant sur une ligne scorée."
        )

    if scored[
        "anomaly_margin"
    ].isna().any():
        raise ValueError(
            "Marge manquante sur une ligne scorée."
        )

    if scored[
        "predicted_is_anomaly"
    ].isna().any():
        raise ValueError(
            "Prédiction manquante sur une ligne scorée."
        )

    expected_prediction = (
        scored["iforest_score"]
        <= scored["decision_threshold"]
    ).astype(int)

    if not np.array_equal(
        expected_prediction.to_numpy(),
        scored[
            "predicted_is_anomaly"
        ].astype(int).to_numpy(),
    ):
        raise ValueError(
            "La règle score/seuil/prédiction "
            "n'est pas respectée."
        )

    expected_margin = (
        scored["decision_threshold"]
        - scored["iforest_score"]
    )

    if not np.allclose(
        scored["anomaly_margin"],
        expected_margin,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "La marge d'anomalie est incorrecte."
        )

    expected_scope = (
        predictions[
            "dataset_split"
        ].eq("test")
        & predictions[
            "operating_state"
        ].eq("RUNNING")
    )

    if not np.array_equal(
        predictions[
            "evaluation_scope"
        ].astype(bool).to_numpy(),
        expected_scope.to_numpy(),
    ):
        raise ValueError(
            "Le périmètre d'évaluation est incorrect."
        )


def compare_metric_value(
    expected: Any,
    actual: Any,
    metric_name: str,
) -> None:
    """Compare deux valeurs de métrique."""

    if expected is None:
        if not pd.isna(actual):
            raise ValueError(
                f"La métrique {metric_name} "
                "devrait être non définie."
            )

        return

    if not np.isclose(
        float(expected),
        float(actual),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Métrique incohérente : {metric_name}."
        )


def validate_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Recalcule et compare les métriques sauvegardées."""

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de métriques introuvable : {METRICS_PATH}"
        )

    saved_metrics = pd.read_csv(
        METRICS_PATH
    )

    overall_rows = saved_metrics[
        saved_metrics["scope"].eq("OVERALL")
    ]

    if len(overall_rows) != 1:
        raise ValueError(
            "La ligne OVERALL est absente ou dupliquée."
        )

    evaluation = predictions[
        predictions[
            "evaluation_scope"
        ].astype(bool)
    ].copy()

    if evaluation.empty:
        raise ValueError(
            "Le périmètre d'évaluation est vide."
        )

    recomputed = recompute_metrics(
        truth=evaluation[
            "ground_truth_is_anomaly"
        ].to_numpy(dtype=int),
        prediction=evaluation[
            "predicted_is_anomaly"
        ].to_numpy(dtype=int),
        anomaly_score=evaluation[
            "anomaly_margin"
        ].to_numpy(dtype=float),
    )

    saved = overall_rows.iloc[0]

    integer_metrics = [
        "evaluated_row_count",
        "actual_anomaly_count",
        "predicted_anomaly_count",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]

    for metric in integer_metrics:
        if int(
            saved[metric]
        ) != int(
            recomputed[metric]
        ):
            raise ValueError(
                f"Métrique entière incohérente : {metric}."
            )

    float_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "false_positive_rate",
        "roc_auc",
        "average_precision",
    ]

    for metric in float_metrics:
        compare_metric_value(
            recomputed[metric],
            saved[metric],
            metric,
        )

    return recomputed


def write_validation_report(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Écrit le rapport final de validation."""

    def format_metric(
        value: Any,
    ) -> str:
        if value is None:
            return "Non défini"

        return f"{float(value):.4f}"

    lines = [
        "RAPPORT DE VALIDATION DU MODÈLE ISOLATION FOREST",
        "=" * 55,
        "Statut : SUCCÈS",
        (
            "Version : "
            f"{manifest['model_version']}"
        ),
        (
            "Nombre de modèles : "
            f"{manifest['model_count']}"
        ),
        (
            "Lignes de la table IA : "
            f"{len(features):,}"
        ),
        (
            "Lignes de prédiction : "
            f"{len(predictions):,}"
        ),
        (
            "Lignes évaluées : "
            f"{metrics['evaluated_row_count']:,}"
        ),
        (
            "Anomalies réelles : "
            f"{metrics['actual_anomaly_count']:,}"
        ),
        (
            "Anomalies prédites : "
            f"{metrics['predicted_anomaly_count']:,}"
        ),
        (
            "Vrais positifs : "
            f"{metrics['true_positive']:,}"
        ),
        (
            "Faux positifs : "
            f"{metrics['false_positive']:,}"
        ),
        (
            "Faux négatifs : "
            f"{metrics['false_negative']:,}"
        ),
        (
            "Précision : "
            f"{format_metric(metrics['precision'])}"
        ),
        (
            "Rappel : "
            f"{format_metric(metrics['recall'])}"
        ),
        (
            "F1-score : "
            f"{format_metric(metrics['f1_score'])}"
        ),
        (
            "ROC-AUC : "
            f"{format_metric(metrics['roc_auc'])}"
        ),
        "",
        "Contrôles réussis :",
        "- un modèle local par équipement",
        "- liste de variables sans étiquette d'évaluation",
        "- conservation de tous les measurement_id",
        "- scores uniquement sur les lignes RUNNING",
        "- cohérence entre score, seuil et prédiction",
        "- périmètre d'évaluation strictement temporel",
        "- recalcul indépendant des métriques",
        "- artefacts modèles lisibles avec joblib",
        "",
        "Limite : résultats obtenus sur un prototype synthétique.",
    ]

    VALIDATION_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main() -> None:
    """Valide les modèles et les résultats."""

    print(
        "Chargement des variables, modèles et prédictions..."
    )

    features = read_dataset(
        FEATURE_DATASET_DIRECTORY
    )

    predictions = read_dataset(
        PREDICTION_DIRECTORY
    )

    for dataframe in [
        features,
        predictions,
    ]:
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="raise",
        )

    manifest = load_json(
        MODEL_MANIFEST_PATH
    )

    feature_payload = load_json(
        FEATURE_LIST_PATH
    )

    feature_columns = list(
        feature_payload[
            "feature_columns"
        ]
    )

    equipment_ids = sorted(
        features[
            "equipment_id"
        ].astype(str).unique()
    )

    validate_feature_integrity(
        features=features,
        predictions=predictions,
    )

    validate_model_files(
        manifest=manifest,
        feature_columns=feature_columns,
        equipment_ids=equipment_ids,
    )

    validate_prediction_rules(
        predictions
    )

    metrics = validate_metrics(
        predictions
    )

    write_validation_report(
        features=features,
        predictions=predictions,
        manifest=manifest,
        metrics=metrics,
    )

    print()
    print(
        "Validation du modèle Isolation Forest réussie."
    )
    print(
        f"Modèles validés : "
        f"{len(equipment_ids)}"
    )
    print(
        f"Lignes de prédiction : "
        f"{len(predictions):,}"
    )
    print(
        f"Lignes évaluées : "
        f"{metrics['evaluated_row_count']:,}"
    )
    print(
        f"Précision : "
        f"{metrics['precision']:.4f}"
        if metrics["precision"] is not None
        else "Précision : non définie"
    )
    print(
        f"Rappel : "
        f"{metrics['recall']:.4f}"
        if metrics["recall"] is not None
        else "Rappel : non défini"
    )
    print(
        f"F1-score : "
        f"{metrics['f1_score']:.4f}"
        if metrics["f1_score"] is not None
        else "F1-score : non défini"
    )
    print(
        f"Rapport : {VALIDATION_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
