import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Tuple, List, Dict, Any, Union
import json
import pickle
import os
from utils.logger import initialize_logger, logger

def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and prepare the dataset.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the features (X), 
        the target (y), and the patient index.
    """
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully.")
    # Separate features and target
    X = df.drop(columns=['patient_index', 'Annotation'])
    y = df['Annotation']
    patient_index = df['patient_index']

    return X, y, patient_index

def custom_stratified_split(X: pd.DataFrame, y: pd.Series, patient_index: pd.Series, n_splits: int = 3, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Perform custom stratified split based on patient indices.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        patient_index (pd.Series): The patient indices.
        n_splits (int, optional): The number of splits for cross-validation. Defaults to 3.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]: A list of tuples containing train and test splits.
    """
    logger.info(f"Performing custom stratified split with {n_splits} splits...")
    logger.info(f"Number of unique patients: {patient_index.nunique()}")
    unique_patients = patient_index.unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)

    # Calculate the distribution of classes for each patient
    patient_classes = [y[patient_index == patient].mode()[0] for patient in unique_patients]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, test_idx in skf.split(unique_patients, patient_classes):
        train_patients, test_patients = unique_patients[train_idx], unique_patients[test_idx]

        train_mask = patient_index.isin(train_patients)
        test_mask = patient_index.isin(test_patients)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        folds.append((X_train, X_test, y_train, y_test))
    logger.info(f"Custom stratified split completed.")
    return folds

def train_and_evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, Union[np.ndarray, None], Dict[str, Any]]:
    """
    Train and evaluate the model.

    Args:
        model (Any): The machine learning model to train.
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.
        X_test (pd.DataFrame): The testing feature matrix.
        y_test (pd.Series): The testing target vector.

    Returns:
        Tuple[pd.Series, pd.Series, Union[np.ndarray, None], Dict[str, Any]]: A tuple containing the true labels, 
        predicted labels, predicted probabilities, and evaluation scores.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    return y_test, y_pred, y_proba, scores

def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series, model_name: str, confusion_matrix_path: str) -> None:
    """
    Plot the confusion matrix.

    Args:
        y_test (pd.Series): The true labels.
        y_pred (pd.Series): The predicted labels.
        model_name (str): The name of the model.
        confusion_matrix_path (str): The path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(confusion_matrix_path)

def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, model_name: str, roc_curve_path: str) -> None:
    """
    Plot the ROC curve.

    Args:
        y_test (pd.Series): The true labels.
        y_proba (np.ndarray): The predicted probabilities.
        model_name (str): The name of the model.
        roc_curve_path (str): The path to save the ROC curve plot.
    """
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f'ROC Curve - {model_name}')
        plt.savefig(roc_curve_path)

def model_pipeline(cfg: DictConfig) -> None:
    """
    Model pipeline function to execute the training and evaluation pipeline.

    Args:
        cfg (DictConfig): The configuration object.
    """
    # Load and prepare data
    X, y, patient_index = load_and_prepare_data(cfg.pipeline.dataset.file_path)

    # Custom stratified split
    folds = custom_stratified_split(X, y, patient_index, cfg.pipeline.dataset.n_splits, cfg.pipeline.dataset.random_state)

    # Define the models with hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=cfg.pipeline.models.random_forest.n_estimators, max_depth=cfg.pipeline.models.random_forest.max_depth,
                                                random_state=cfg.pipeline.models.random_forest.random_state),
        'XGBoost': xgb.XGBClassifier(n_estimators=cfg.pipeline.models.xgboost.n_estimators, max_depth=cfg.pipeline.models.xgboost.max_depth,
                                      learning_rate=cfg.pipeline.models.xgboost.learning_rate, random_state=cfg.pipeline.models.xgboost.random_state),
        'LightGBM': lgb.LGBMClassifier(n_estimators=cfg.pipeline.models.lightgbm.n_estimators, max_depth=cfg.pipeline.models.lightgbm.max_depth,
                                       learning_rate=cfg.pipeline.models.lightgbm.learning_rate, random_state=cfg.pipeline.models.lightgbm.random_state)
    }

    # Train and evaluate models
    results = {model_name: {'accuracy': [], 'classification_report': [], 'roc_auc': []} for model_name in models}

    logger.info("Training and evaluating models...")
    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
        logger.info(f"Fold {fold_idx + 1}/{cfg.pipeline.dataset.n_splits}")
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            print(f"Training {model_name}...")
            y_test_fold, y_pred_fold, y_proba_fold, fold_scores = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            logger.info(f"Evaluation scores for {model_name}: {fold_scores}")
            for metric in fold_scores:
                if fold_scores[metric] is not None:
                    results[model_name][metric].append(fold_scores[metric])
            logger.info(f"Model {model_name} trained and evaluated successfully.")
            # Save model weights
            logger.info(f"Saving model weights for {model_name}...")
            if not os.path.exists(cfg.pipeline.models.save_path):
                os.makedirs(cfg.pipeline.models.save_path, exist_ok=True)
            model_path = f"{cfg.pipeline.models.save_path}_{model_name}_fold_{fold_idx + 1}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model weights saved successfully at {model_path}.")
            logger.info(f"Plotting confusion matrix and ROC curve for {model_name}...")
            # Plot confusion matrix and ROC curve for each fold
            plot_confusion_matrix(y_test_fold, y_pred_fold, model_name, f"{cfg.pipeline.confusion_matrix.path}_{model_name}_fold_{fold_idx + 1}.png")
            plot_roc_curve(y_test_fold, y_proba_fold, model_name, f"{cfg.pipeline.roc_curve.path}_{model_name}_fold_{fold_idx + 1}.png")
            logger.info(f"Confusion matrix and ROC curve plots saved successfully.")
    logger.info("Training and evaluation completed.")
    logger.info("Calculating mean metrics...")
    logger.info("Mean Metrics:")
    # Save the results
    final_results = {}
    for model_name, metrics in results.items():
        mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items() if scores and metric != 'classification_report'}
        mean_metrics['classification_report'] = metrics['classification_report']
        logger.info(f"{model_name}: {mean_metrics}")

    logger.info("Mean metrics calculated successfully.")
    logger.info("Saving results...")
    with open(cfg.pipeline.results.path, 'w') as f:
        json.dump(final_results, f, indent=4)
    logger.info(f"Results saved successfully at {cfg.pipeline.results.path}.")