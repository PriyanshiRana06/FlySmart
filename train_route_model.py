
import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "domestic_fares_clean.csv"
MODEL_PATH = BASE_DIR / "model" / "route_model.joblib"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "origin",
    "destination"
]

NUMERICAL_FEATURES = [
    "observations"
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# ==========================================
# ARGUMENTS
# ==========================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train FlySmart route fare classifier"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10
    )

    return parser.parse_args()


# ==========================================
# PREPROCESSING
# ==========================================

def create_pipeline(classifier):

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    preprocessing = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        )
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
    ])


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y, predictions, average="macro", zero_division=0
        )
    }


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    args = parse_arguments()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["median"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    df = df.dropna(subset=required_columns).copy()

    # Create target using historical median fare.
    # Median is deliberately excluded from model inputs.

    df["fare_category"] = pd.qcut(
        df["median"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    X = df[FEATURES]
    y = df["fare_category"].astype(str)

    print("\nDataset shape:", df.shape)
    print("\nFeatures:", FEATURES)

    print("\nCategory distribution:")
    print(y.value_counts())

    # 60% training, 20% validation, 20% testing.

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=args.random_state,
            stratify=y
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_development,
            y_development,
            test_size=0.25,
            random_state=args.random_state,
            stratify=y_development
        )
    )

    print("\nData split:")
    print("Training:", len(X_train))
    print("Validation:", len(X_validation))
    print("Testing:", len(X_test))

    # ======================================
    # MLFLOW CONFIGURATION
    # ======================================

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "FlySmart_Route_Fare_Classification"
    )

    # ======================================
    # MODELS
    # ======================================

    models = {

        "Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            random_state=args.random_state
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    results = []

    # ======================================
    # TRAIN AND TRACK EXPERIMENTS
    # ======================================

    for model_name, classifier in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_pipeline(classifier)

        with mlflow.start_run(run_name=model_name):

            pipeline.fit(X_train, y_train)

            validation_metrics = evaluate_model(
                pipeline,
                X_validation,
                y_validation
            )

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "features",
                str(FEATURES)
            )

            mlflow.log_param(
                "random_state",
                args.random_state
            )

            if model_name == "Random_Forest":
                mlflow.log_param(
                    "n_estimators",
                    args.n_estimators
                )

                mlflow.log_param(
                    "max_depth",
                    args.max_depth
                )

            mlflow.log_metrics(validation_metrics)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="route_fare_classifier",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )

            results.append({
                "name": model_name,
                "model": pipeline,
                "validation_f1": validation_metrics["f1_macro"]
            })

            print("Validation metrics:", validation_metrics)

    # ======================================
    # SELECT BEST MODEL
    # ======================================

    best_result = max(
        results,
        key=lambda result: result["validation_f1"]
    )

    best_model = best_result["model"]

    print("\nBest model:", best_result["name"])

    # ======================================
    # FINAL TEST EVALUATION
    # ======================================

    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            best_model.predict(X_test),
            zero_division=0
        )
    )

    # ======================================
    # SAVE MODEL SEPARATELY
    # ======================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(best_model, MODEL_PATH)

    print("\nRoute model saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
