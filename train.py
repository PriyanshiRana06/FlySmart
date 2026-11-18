
import os
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/Dataset.csv"
MODEL_PATH = "model/model.joblib"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]

NUMERICAL_FEATURES = [
    "duration",
    "days_left"
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET = "Price_Category"


# --------------------------------------------------
# 2. COMMAND-LINE ARGUMENTS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train FlySmart flight price classification models"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=20,
        help="Maximum depth of Random Forest trees"
    )

    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="Minimum samples required in a leaf"
    )

    parser.add_argument(
        "--lr_c1",
        type=float,
        default=1.0,
        help="Regularization parameter for first Logistic Regression"
    )

    parser.add_argument(
        "--lr_c01",
        type=float,
        default=0.1,
        help="Regularization parameter for second Logistic Regression"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.n_estimators < 1:
        parser.error("--n_estimators must be at least 1")

    if args.max_depth < 1:
        parser.error("--max_depth must be at least 1")

    if args.min_samples_leaf < 1:
        parser.error("--min_samples_leaf must be at least 1")

    if args.lr_c1 <= 0 or args.lr_c01 <= 0:
        parser.error("Logistic Regression C values must be positive")

    return args


# --------------------------------------------------
# 3. LOAD AND PREPARE DATASET
# --------------------------------------------------

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["price"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Create three price categories
    df[TARGET] = pd.qcut(
        df["price"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str)

    print("\nDataset shape:", df.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nInput features:", FEATURES)

    return X, y


# --------------------------------------------------
# 4. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

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

    preprocessor = ColumnTransformer([
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

    return preprocessor


# --------------------------------------------------
# 5. CALCULATE EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, predictions):

    return {
        "accuracy": accuracy_score(y_true, predictions),

        "precision_macro": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# --------------------------------------------------
# 6. TRAIN AND TRACK ONE EXPERIMENT
# --------------------------------------------------


def run_experiment(
    model_name,
    classifier,
    X_train,
    X_val,
    y_train,
    y_val,
    parameters,
    random_state
):
    print(f"\nTraining: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    with mlflow.start_run(run_name=model_name) as run:

        # 1. Log experiment parameters
        mlflow.log_params(parameters)

        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("features", str(FEATURES))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("random_state", random_state)

        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))

        # 2. Train the model
        print("Fitting model...")

        pipeline.fit(X_train, y_train)

        print("Training completed!")

        # 3. Predict on validation data
        predictions = pipeline.predict(X_val)

        # 4. Calculate validation metrics
        metrics = calculate_metrics(y_val, predictions)

        # 5. Log validation metrics in MLflow
        mlflow.log_metrics({
            f"validation_{name}": value
            for name, value in metrics.items()
        })

        # 6. Log trained model in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="flight_price_classifier",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        # 7. Display validation results
        print("\nValidation Results:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # 8. Display classification report
        print("\nClassification Report:")

        print(
            classification_report(
                y_val,
                predictions,
                zero_division=0
            )
        )

        # 9. Return experiment details
        return {
            "name": model_name,
            "pipeline": pipeline,
            "validation_f1": metrics["f1_macro"],
            "run_id": run.info.run_id
        }

# --------------------------------------------------
# 7. EVALUATE BEST MODEL ON TEST DATA
# --------------------------------------------------

def evaluate_best_model(best_model, X_test, y_test):

    print("\n========== FINAL TEST EVALUATION ==========")

    predictions = best_model["pipeline"].predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    print("\nFinal Test Metrics:")

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    print("\nFinal Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Add final test metrics to the selected MLflow run

    with mlflow.start_run(run_id=best_model["run_id"]):

        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in metrics.items()
        })

    return metrics


# --------------------------------------------------
# 8. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    args = parse_arguments()

    os.makedirs("model", exist_ok=True)

    # Configure MLflow SQLite tracking

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "FlySmart_Flight_Classification"
    )

    # Load dataset

    X, y = load_data()

    # First split: 80% development, 20% test

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=args.random_state,
            stratify=y
        )
    )

    # Second split: 60% training, 20% validation

    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=args.random_state,
        stratify=y_development
    )

    print("\n========== DATA SPLIT ==========")

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))


    # --------------------------------------------------
    # EXPERIMENT 1: LOGISTIC REGRESSION C=1.0
    # --------------------------------------------------

    experiment_1 = run_experiment(
        "Logistic_Regression_C1",

        LogisticRegression(
            C=args.lr_c1,
            max_iter=1000,
            random_state=args.random_state
        ),

        X_train,
        X_val,
        y_train,
        y_val,

        {
            "C": args.lr_c1,
            "max_iter": 1000
        },

        args.random_state
    )


    # --------------------------------------------------
    # EXPERIMENT 2: LOGISTIC REGRESSION C=0.1
    # --------------------------------------------------

    experiment_2 = run_experiment(
        "Logistic_Regression_C01",

        LogisticRegression(
            C=args.lr_c01,
            max_iter=1000,
            random_state=args.random_state
        ),

        X_train,
        X_val,
        y_train,
        y_val,

        {
            "C": args.lr_c01,
            "max_iter": 1000
        },

        args.random_state
    )


    # --------------------------------------------------
    # EXPERIMENT 3: RANDOM FOREST
    # --------------------------------------------------

    experiment_3 = run_experiment(
        "Random_Forest",

        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
            random_state=args.random_state
        ),

        X_train,
        X_val,
        y_train,
        y_val,

        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": -1
        },

        args.random_state
    )


    # --------------------------------------------------
    # 9. COMPARE EXPERIMENTS
    # --------------------------------------------------

    experiments = [
        experiment_1,
        experiment_2,
        experiment_3
    ]

    print("\n========== MODEL COMPARISON ==========")

    for result in experiments:

        print(
            f"{result['name']}: "
            f"Validation F1 = {result['validation_f1']:.4f}"
        )

    # Select using validation F1 only

    best_model = max(
        experiments,
        key=lambda result: result["validation_f1"]
    )

    print("\nSelected Model:", best_model["name"])

    print(
        "Validation F1:",
        round(best_model["validation_f1"], 4)
    )


    # --------------------------------------------------
    # 10. FINAL TEST EVALUATION
    # --------------------------------------------------

    final_metrics = evaluate_best_model(
        best_model,
        X_test,
        y_test
    )


    # --------------------------------------------------
    # 11. SAVE BEST MODEL
    # --------------------------------------------------

    joblib.dump(
        best_model["pipeline"],
        MODEL_PATH
    )

    print("\n========== TRAINING COMPLETE ==========")

    print("Best model:", best_model["name"])

    print("Final Test Accuracy:", round(
        final_metrics["accuracy"], 4
    ))

    print("Final Test F1:", round(
        final_metrics["f1_macro"], 4
    ))

    print("Saved model to:", MODEL_PATH)


# --------------------------------------------------
# 12. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
