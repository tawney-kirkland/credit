import pandas as pd
import numpy as np
import joblib
import wandb
import snowflake.connector
import json
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import logging

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Function to load the pre-trained model and the pre-processing pipeline
def load_model_and_preprocessor(model_path="model_artifacts/random_forest_model_feature_engineered.pkl"):
    model_pipeline = joblib.load(model_path)
    return model_pipeline

# Function to get the last processed ID from a JSON file
# Since we don't have a timestamp field in the table
def get_last_processed_id(file_path="last_processed_id.json"):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            file.close()
        return data.get("last_processed_id", 0)  # Default to 0 if not found
        
    except FileNotFoundError:
        return 0  # Default to 0 if the file does not exist
    except json.JSONDecodeError:
        logger.warning("Error reading JSON, returning 0 as default.")
        return 0  # Return 0 if the file content is not valid JSON

# Function to update the last processed ID after testing
def update_last_processed_id(file_path="last_processed_id.json",new_last_processed_id=0):
    try:
        logger.info(f"Attempting to write to {file_path}")

        # Convert to python int before writing to JSON
        new_last_processed_id = int(new_last_processed_id)
        
        # Open the file and write the new last processed ID
        with open(file_path, "w") as file:
            # Create the data to be written
            data = {"last_processed_id": new_last_processed_id}
            json.dump(data, file)
            logger.info(f"Successfully updated last_processed_id to {new_last_processed_id}.")
            
    except OSError as e:
        logger.error(f"Error updating last_processed_id: {e}")
        raise  # Re-raise the error to halt execution or handle it further

# Fetch new data from Snowflake
def fetch_new_data(last_processed_id, limit=2000):
    conn = snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )

    # Query to get the next batch of data from Snowflake
    query = f"""
    SELECT * FROM credit.dev.dim_credit_cleansed
    WHERE UNIQUE_ID > {last_processed_id}
    LIMIT {limit}
    """
   
    return pd.read_sql(query, conn)

# Feature engineering (optional)
def feature_engineering(df, cap_values=True):

    # Ensure the columns involved are numeric (if they are not, coerce them)
    df['DEBT_RATIO'] = pd.to_numeric(df['DEBT_RATIO'], errors='coerce')
    df['COUNT_90_DAYS_LATE'] = pd.to_numeric(df['COUNT_90_DAYS_LATE'], errors='coerce')
    df['CREDIT_USAGE_PCT'] = pd.to_numeric(df['CREDIT_USAGE_PCT'], errors='coerce')
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

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

# Function to evaluate the model on new data
def test_new_data(model_pipeline, new_data):

    # Apply the feature engineering to the new data
    new_data = feature_engineering(new_data)
    
    # Extract target variable for the new data
    y_test = new_data['SERIOUS_DELINQUENCY']
    X_test = new_data.drop(['SERIOUS_DELINQUENCY', 'UNIQUE_ID','COUNT_6089_DAYS_PAST_DUE','COUNT_3059_DAYS_PAST_DUE'], axis=1)
    print(X_test.columns)
    # Apply the same preprocessing (e.g., imputation, scaling) to the new data
    new_data_processed = model_pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Get predictions and probabilities from the model
    y_pred = model_pipeline.predict(new_data_processed)
    y_prob = model_pipeline.predict_proba(new_data_processed)

    # Evaluate model performance
    auc = roc_auc_score(y_test, y_prob[:, 1])
    logger.info(f"AUC: {auc:.4f}")
    logger.info("Classification Report:\n", classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Log the results to W&B
    wandb.log({"AUC": auc, "classification_report": classification_report(y_test, y_pred)})
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred)
    wandb.sklearn.plot_roc(y_test, np.vstack([1 - y_prob[:, 1], y_prob[:, 1]]).T )
    wandb.sklearn.plot_precision_recall(y_test, y_prob)

def main():
    # Initialize W&B
    # Initialize W&B
    wandb.init(project="credit-risk-model", job_type="model_testing")

    # Load the pre-trained model pipeline
    model_pipeline = load_model_and_preprocessor()

    # Get the last processed ID
    last_processed_id = get_last_processed_id()
    new_data = fetch_new_data(last_processed_id)
    print(new_data.head(10))
    # Test the model on new data
    test_new_data(model_pipeline, new_data)


    # Get the max ID from the new data and update last_processed_id
    new_last_processed_id = new_data['UNIQUE_ID'].max()  
    logger.info(f"New last_processed_id: {new_last_processed_id}")
    update_last_processed_id("last_processed_id.json", new_last_processed_id)

# Run the pipeline
if __name__ == '__main__':
    main()
