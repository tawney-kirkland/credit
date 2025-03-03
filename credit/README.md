# Credit Risk Prediction

This project walks through the full Machine Learning (ML) lifecycle with the goal of predicting the probability that an individual will experience financial distress in the next two years. The dataset used for this analysis comes from the Kaggle [Give Me Some Credit competition](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview).

## Project Overview

The goal of this project is to predict the likelihood of financial distress for individuals based on various financial and demographic factors. This is framed as a binary classification problem where the objective is to predict whether a person will experience financial distress within two years.

1. Data collection and ETL
- The data pipeline is orchestrated through dbt (Data Build Tool), with data stored in Snowflake for easy querying and transformations.
- The dataset undergoes Extract, Load, and Transform (ELT) to ensure it is ready for modeling.

2. Modeling
- The model training begins with a Random Forest baseline, as seen in `model_training.py`. The model is trained and validated, and its performance is monitored throughout the process.
- Models and runs are logged in Weights and Biases to monitor ongoing model performance.

3. Next steps
- The model will be improved with XGBoost and other algorithms.
- Future work includes automating the testing pipeline and deploying the model for use with new data.