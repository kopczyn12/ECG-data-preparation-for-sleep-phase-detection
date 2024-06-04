import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Tuple, Dict, Any, Union
import json
import pickle
import os
from utils.logger import initialize_logger, logger
from sklearn.utils import resample

# Define the inverse mapping
inverse_annotation_mapping = {
    0: 'W',
    1: 'R',
    2: 'Deep'
}

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
    logger.info("Data loaded successfully.")

    # Mapping for annotation column with merged N1, N2, N3 into a single class
    annotation_mapping = {
        ' W': 0,  # Wake
        ' R': 1,  # REM
        ' N1': 2, # Non-REM (merged N1, N2, N3)
        ' N2': 2, # Non-REM (merged N1, N2, N3)
        ' N3': 2  # Non-REM (merged N1, N2, N3)
    }

    # Separate features and target
    X = df.drop(columns=['patient_index', 'Annotation', 'epoch_index', 'MCVNN'])
    y = df['Annotation'].map(annotation_mapping)

    patient_index = df['patient_index']

    # Remove rows with NaN values in target
    non_nan_mask = y.notna()
    X = X[non_nan_mask]
    y = y[non_nan_mask]
    patient_index = patient_index[non_nan_mask]

    # Combine features and target for upsampling
    df_upsample = pd.concat([X, y, patient_index], axis=1)

    # Separate majority and minority classes
    df_majority = df_upsample[df_upsample['Annotation'] == 2]
    df_minority_0 = df_upsample[df_upsample['Annotation'] == 0]
    df_minority_1 = df_upsample[df_upsample['Annotation'] == 1]

    # Upsample minority classes to 50% of the majority class size
    target_samples = int(len(df_majority) * 0.5)

    df_minority_0_upsampled = resample(df_minority_0,
                                       replace=True,    # sample with replacement
                                       n_samples=target_samples, # to half of majority class
                                       random_state=123) # reproducible results

    df_minority_1_upsampled = resample(df_minority_1,
                                       replace=True,
                                       n_samples=target_samples,
                                       random_state=123)

    # Combine majority class with upsampled minority classes
    df_upsampled = pd.concat([df_majority, df_minority_0_upsampled, df_minority_1_upsampled])

    # Separate features and target again
    X_upsampled = df_upsampled.drop(columns=['patient_index', 'Annotation'])
    y_upsampled = df_upsampled['Annotation']
    patient_index_upsampled = df_upsampled['patient_index']

    return X_upsampled, y_upsampled, patient_index_upsampled

def split_data(X: pd.DataFrame, y: pd.Series, patient_index: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets without overlapping patients.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        patient_index (pd.Series): The patient indices.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the train-test split of features and target.
    """
    logger.info(f"Splitting data with test size {test_size} and random state {random_state}...")

    # Get unique patients
    unique_patients = patient_index.unique()

    # Split patients into train and test sets
    train_patients, test_patients = train_test_split(unique_patients, test_size=test_size, random_state=random_state)

    # Create masks for train and test sets
    train_mask = patient_index.isin(train_patients)
    test_mask = patient_index.isin(test_patients)

    # Split the data
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Data splitting completed.")

    return X_train, X_test, y_train, y_test

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

    # Convert predictions and true labels back to original string values
    y_test_str = y_test.map(inverse_annotation_mapping)
    y_pred_str = pd.Series(y_pred).map(inverse_annotation_mapping)

    scores = {
        'accuracy': accuracy_score(y_test_str, y_pred_str),
        'classification_report': classification_report(y_test_str, y_pred_str, output_dict=True),
    }
    logger.info(f"Accuracy: {scores['accuracy']}")
    logger.info(f"Classification Report:\n{scores['classification_report']}")
    return y_test_str, y_pred_str, y_proba, scores

def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series, model_name: str, confusion_matrix_path: str) -> None:
    """
    Plot the confusion matrix.

    Args:
        y_test (pd.Series): The true labels.
        y_pred (pd.Series): The predicted labels.
        model_name (str): The name of the model.
        confusion_matrix_path (str): The path to save the confusion matrix plot.
    """
    # Convert predictions and true labels back to original string values if they are not already
    if y_test.dtype != 'object':
        y_test = y_test.map(inverse_annotation_mapping)
    if y_pred.dtype != 'object':
        y_pred = y_pred.map(inverse_annotation_mapping)

    cm = confusion_matrix(y_test, y_pred, labels=['W', 'R', 'Deep'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['W', 'R', 'Deep'])
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(confusion_matrix_path)
    plt.close()

def plot_feature_importance(importance, names, model_type, save_path):
    """
    Create and save a plot for feature importance.

    Args:
        importance (array): Feature importance values.
        names (array): Feature names.
        model_type (str): Type of the model.
        save_path (str): Path to save the plot.
    """
    # Create a DataFrame for visualization
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'Feature Names': feature_names, 'Feature Importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame by feature importance
    fi_df.sort_values(by=['Feature Importance'], ascending=False, inplace=True)

    # Plot the feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['Feature Names'], fi_df['Feature Importance'])
    plt.title(f'Feature Importance for {model_type}')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def model_pipeline(cfg: DictConfig) -> None:
    """
    Model pipeline function to execute the training and evaluation pipeline.

    Args:
        cfg (DictConfig): The configuration object.
    """
    # Load and prepare data
    X, y, patient_index = load_and_prepare_data(cfg.pipeline.dataset.file_path)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y, patient_index, test_size=cfg.pipeline.dataset.test_size, random_state=cfg.pipeline.dataset.random_state)

    # Define the models with hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=cfg.pipeline.models.random_forest.n_estimators, max_depth=cfg.pipeline.models.random_forest.max_depth,
                                                random_state=cfg.pipeline.models.random_forest.random_state),
        'XGBoost': xgb.XGBClassifier(n_estimators=cfg.pipeline.models.xgboost.n_estimators, max_depth=cfg.pipeline.models.xgboost.max_depth,
                                      learning_rate=cfg.pipeline.models.xgboost.learning_rate, random_state=cfg.pipeline.models.xgboost.random_state)
    }

    # Train and evaluate models
    results = {model_name: {'accuracy': [], 'classification_report': [], 'roc_auc': []} for model_name in models}

    logger.info("Training and evaluating models...")
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        print(f"Training {model_name}...")
        y_test_str, y_pred_str, y_proba, scores = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        logger.info(f"Evaluation scores for {model_name}: {scores}")
        for metric in scores:
            if scores[metric] is not None:
                results[model_name][metric].append(scores[metric])
        logger.info(f"Model {model_name} trained and evaluated successfully.")

        # Save model weights
        logger.info(f"Saving model weights for {model_name}...")
        if not os.path.exists(cfg.pipeline.models.save_path):
            os.makedirs(cfg.pipeline.models.save_path, exist_ok=True)
        model_path = os.path.join(cfg.pipeline.models.save_path, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model weights saved successfully at {model_path}.")

        # Plot confusion matrix for the model
        logger.info(f"Plotting confusion matrix for {model_name}...")
        if not os.path.exists(cfg.pipeline.confusion_matrix.path):
            os.makedirs(cfg.pipeline.confusion_matrix.path, exist_ok=True)
        plot_confusion_matrix(y_test_str, y_pred_str, model_name, os.path.join(cfg.pipeline.confusion_matrix.path, f"{model_name}.png"))
        logger.info(f"Confusion matrix plot saved successfully.")

        # Plot feature importance for the model
        logger.info(f"Plotting feature importance for {model_name}...")
        if not os.path.exists(cfg.pipeline.feature_importance.path):
            os.makedirs(cfg.pipeline.feature_importance.path, exist_ok=True)

        if model_name == 'RandomForest':
            importances = model.feature_importances_
        elif model_name == 'XGBoost':
            importances = model.get_booster().get_score(importance_type='weight')
            importances = [importances.get(f'f{i}', 0.0) for i in range(X_train.shape[1])]

        if not os.path.exists(cfg.pipeline.feature_importance_path):
            os.makedirs(cfg.pipeline.feature_importance_path, exist_ok=True)

        plot_feature_importance(importances, X_train.columns, model_name, os.path.join(cfg.pipeline.feature_importance_path, f"{model_name}_feature_importance.png"))
        logger.info(f"Feature importance plot saved successfully for {model_name}.")

    logger.info("Training and evaluation completed.")
    logger.info("Calculating mean metrics...")
    logger.info("Mean Metrics:")
    # Save the results
    final_results = {}
    if not os .path.exists(cfg.pipeline.results_path):
        os.makedirs(cfg.pipeline.results_path, exist_ok=True)
    for model_name, metrics in results.items():
        mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items() if scores and metric != 'classification_report'}
        mean_metrics['classification_report'] = metrics['classification_report']
        logger.info(f"{model_name}: {mean_metrics}")
        final_results[model_name] = mean_metrics

        logger.info("Saving results...")
        with open(f"{cfg.pipeline.results_path}_results_{model_name}.json", 'w') as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Results saved successfully at {cfg.pipeline.results_path}.")
    logger.info("Mean metrics calculated and saved successfully.")
