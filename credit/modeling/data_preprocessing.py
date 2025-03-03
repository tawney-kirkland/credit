import os
import snowflake.connector
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from dotenv import load_dotenv

load_dotenv()

# Setup Snowflake connection
def get_snowflake_data():
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )
    
    # Query Snowflake for training data 
    query = "SELECT * FROM credit.dev.dim_credit_cleansed LIMIT 100000"
    
    # Retrieve data into a Pandas DataFrame
    df = pd.read_sql(query, conn)
    
    conn.close()
    
    return df

# Function for preprocessing
# Currently limited to imputing missing values
def preprocess_data(df):

    # Drop strongly correlated attributes
    df = df.drop(['COUNT_6089_DAYS_PAST_DUE', 'COUNT_3059_DAYS_PAST_DUE'], axis=1)

    # Initialize the imputer (replace missing values with the mean)
    imputer = SimpleImputer(strategy='mean')
    
    # Apply the imputer to the entire DataFrame
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    joblib.dump(imputer, 'imputer.pkl')
    
    return df_imputed

# Save the preprocessed data to a Parquet file
def save_data_to_parquet(df, file_name='datasets/preprocessed_data.parquet'):
    df.to_parquet(file_name, index=False)
    print(f"Data saved to {file_name}")

def main():
    df = get_snowflake_data()
    df_imputed = preprocess_data(df)
    save_data_to_parquet(df_imputed) 

if __name__ == '__main__':
    main()