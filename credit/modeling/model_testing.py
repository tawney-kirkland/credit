import pandas as pd
import joblib
import wandb
import snowflake.connector
import json
from sklearn.metrics import classification_report, roc_auc_score
import os
import logging

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Function to load the pre-trained model
def load_model(model_path="model_artifacts/random_forest_model.pkl"):
    return joblib.load(model_path)

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
        
        # Open the file and write the new last processed ID
        with open("last_processed_id.json", "w") as file:
            # Create the data to be written
            data = {"last_processed_id": new_last_processed_id}
            json.dump(data, file)
            logger.info("Successfully updated the last_processed_id.")
            
    except OSError as e:
        logger.error(f"Error updating last_processed_id: {e}")
        raise  # Re-raise the error to halt execution or handle it further

# Function to fetch new data from Snowflake (replace with actual Snowflake connection logic)
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
    # Assuming you have a function to execute the query and return data as DataFrame
    return pd.read_sql(query, conn)  # Replace with actual Snowflake connection logic

# Function to evaluate the model on new data
def test_new_data(model, new_data):
    # Fetch the new data
    # new_data = fetch_new_data(last_processed_id)

    # Split data into features and target
    X_test = new_data.drop(['SERIOUS_DELINQUENCY', 'COUNT_6089_DAYS_PAST_DUE', 'COUNT_3059_DAYS_PAST_DUE','UNIQUE_ID'], axis=1)
    y_test = new_data['SERIOUS_DELINQUENCY']

    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1

    # Evaluate model performance
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Log the results to W&B
    wandb.log({"AUC": auc, "classification_report": classification_report(y_test, y_pred)})


# Main function to run the testing pipeline
def main():
    # Initialize W&B
    wandb.init(project="credit-risk-model", job_type="model_testing")

    # Load the pre-trained model
    model = load_model()

    # Get the last processed ID
    last_processed_id = get_last_processed_id()
    new_data = fetch_new_data(last_processed_id)
    
    # Test the model on new data
    test_new_data(model, new_data)
    
    # Get the max ID from the new data and update last_processed_id
    new_last_processed_id = new_data['UNIQUE_ID'].max()  
    logger.info(f"New last_processed_id: {new_last_processed_id}")
    update_last_processed_id(new_last_processed_id)
    
# Run the pipeline
if __name__ == '__main__':
    main()
