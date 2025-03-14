import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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

# Feature engineering (optional)
def feature_engineering(df, cap_values=True):
    if cap_values:
        df['DEBT_RATIO'] = np.where(df['DEBT_RATIO'] > 10, 10, df['DEBT_RATIO'])
        df['COUNT_90_DAYS_LATE'] = np.where(df['COUNT_90_DAYS_LATE'] > 20,
                                                  df['COUNT_90_DAYS_LATE'].mean(), df['COUNT_90_DAYS_LATE'])
        df['CREDIT_USAGE_PCT'] = np.where(df['CREDIT_USAGE_PCT'] > 2,
                                                               2, df['CREDIT_USAGE_PCT'])
        df['AGE'] = np.where(df['AGE'] > 85, 85, df['AGE'])

    # Create the 'group_90days_late' feature
    df['group_90days_late'] = np.where(df['COUNT_90_DAYS_LATE'] == 0, '0', 
                                       np.where((df['COUNT_90_DAYS_LATE'] >= 1) & 
                                                (df['COUNT_90_DAYS_LATE'] <= 4), '1-4', '5+'))
    df['credit_usage_age'] = df['AGE'] * df['CREDIT_USAGE_PCT']
    df.drop(columns=['COUNT_90_DAYS_LATE'], inplace=True)

    return df

# Train the model
## NEED TO CONTINUE UPDATING FOR FLEXIBILITY / REUSABILITY
## ie FIX feature_engineering, specifically for preprocessing
def train_model(df, feature_engineering_fn=feature_engineering, model=RandomForestClassifier(), test_size=0.25, random_state=12, log=True):
    # Optionally apply feature engineering
    df = feature_engineering_fn(df)

    X = df.drop(['SERIOUS_DELINQUENCY', 'UNIQUE_ID','COUNT_3059_DAYS_PAST_DUE','COUNT_6089_DAYS_PAST_DUE'], axis=1)
    y = df['SERIOUS_DELINQUENCY']
    print(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # # Create the Random Forest pipeline
    # model_pipeline = Pipeline(steps=[
    #     ('classifier', model)
    # ])

    # Preprocessing for numerical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Preprocessing pipeline for numerical features (impute missing with the mean)
    numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Impute missing values with the mean
    ])

    # Preprocessing pipeline for categorical features (one-hot encode)
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))  # One-hot encode the categories
    ])

    # Combine both pipelines into a single column transformer
    # (dropped scaling but may re integrate later)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Create the Random Forest pipeline
    # Used class_weight='balanced' to account for the class imbalance. 
    # This will give more weight to the minority class (delinquent). 
    # This helps to improve recall (i.e., identifying more delinquent cases) but at the cost of precision (increased false positives).
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,class_weight='balanced'))
    ])

    # Train the model
    logger.info("Training the model...")
    model_pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)  # Get probabilities for the positive class (delinquent)

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
    # Load the preprocessed data
    df = load_preprocessed_data()

    # Initialize W&B
    wandb.init(project="credit-risk-model", job_type="model_training")

    # Choose the model (RandomForest, XGBoost, etc.) and other configurations
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=12, class_weight='balanced')

    # Train the model with feature engineering and logging enabled
    model_pipeline = train_model(df, feature_engineering_fn=feature_engineering, model=model)

    # Save the model and log it to W&B
    model_path = "model_artifacts/random_forest_model_feature_engineered.pkl"
    joblib.dump(model_pipeline, model_path)
    wandb.save(model_path)

    logger.info(f"Model saved and logged to W&B at {model_path}")

if __name__ == '__main__':
    main()

