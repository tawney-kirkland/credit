import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import wandb
import os
import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Initialize Wandb
wandb.init(project="credit-risk-model", config={
    "n_estimators": 100,
    "max_depth": 10,
    "test_size": 0.25,
    "random_state": 12,
})

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load the preprocessed data
def load_preprocessed_data(file_name='datasets/preprocessed_data.parquet'):
    return pd.read_parquet(file_name)

def save_and_log_datasets(train_df, test_df, y_probas, run):
    """
    Save the split datasets and probabilities to CSV files and log them as artifacts in W&B.
    """
    os.makedirs('datasets', exist_ok=True)

    train_df.to_csv('datasets/train.csv', index=False)
    test_df.to_csv('datasets/test.csv', index=False)

    artifact = wandb.Artifact(name="credit-dataset", type="dataset")
    artifact.add_file(local_path="datasets/train.csv", name="train.csv")
    artifact.add_file(local_path="datasets/test.csv", name="test.csv")
    
    run.log_artifact(artifact)
    logger.info("Datasets saved and logged as artifacts in W&B.")

# Train the model
def train_model():
    df = load_preprocessed_data()
    print(df.head())
    X = df.drop(['SERIOUS_DELINQUENCY', 'UNIQUE_ID'], axis=1)
    y = df['SERIOUS_DELINQUENCY']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=wandb.config.test_size, random_state=wandb.config.random_state)

    # Create the Random Forest pipeline
    model_pipeline = Pipeline(steps=[
        ('classifier', RandomForestClassifier(
            n_estimators=wandb.config.n_estimators,
            max_depth=wandb.config.max_depth,
            random_state=wandb.config.random_state,
            class_weight='balanced'))
    ])

    # Train the model
    logger.info("Training the model...")
    model_pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test) # Get probabilities for the positive class (delinquent)
    # y_prob_reshaped = np.vstack([1 - y_prob, y_prob]).T
    print(f"y_prob shape: {y_prob.shape}")
    # Calculate the AUC
    auc = roc_auc_score(y_test, y_prob[:, 1])
    logger.info(f"AUC: {auc:.4f}")
    wandb.log({"AUC": auc})

    # Evaluate the model
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    wandb.log({"classification_report": classification_report(y_test, y_pred)})

    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    wandb.log({"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()})

    # Cross-validation performance
    cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    wandb.log({"cv_accuracy": cv_scores.mean()})

    wandb.sklearn.plot_confusion_matrix(y_test, y_pred)
    wandb.sklearn.plot_roc(y_test, np.vstack([1 - y_prob[:, 1], y_prob[:, 1]]).T )

    wandb.sklearn.plot_precision_recall(y_test, y_prob)
    wandb.sklearn.plot_class_proportions(y_train, y_test)

    # Retrain on the full dataset
    logger.info("Retraining the model on the full dataset...")
    model_pipeline.fit(X, y)

    return model_pipeline


def main():

    model_pipeline = train_model()

    # Save the model and log it to W&B
    model_path = "model_artifacts/random_forest_model.pkl"
    joblib.dump(model_pipeline, model_path)
    wandb.save(model_path)

    logger.info(f"Model saved and logged to W&B at {model_path}")

if __name__ == '__main__':
    main()
