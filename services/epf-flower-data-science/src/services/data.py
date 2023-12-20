from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import joblib
import os

def download_iris_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='services/epf-flower-data-science/src/data', unzip=True)

    return {"status": "Dataset downloaded correctly"}

def load_iris_dataset():
    file_path = 'services/epf-flower-data-science/src/data/Iris.csv'
    try:
        df = pd.read_csv(file_path)
        return df.to_json(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}

def processing_dataset():
    data = load_iris_dataset()
    # file_path = 'services/epf-flower-data-science/src/data/Iris.csv'
    try:
        df = pd.read_json(data)
        df['Species'] = df['Species'].apply(lambda x: x.replace('Iris-', ''))  
        return df.to_json(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}
    
def split_dataset():
    dataset_processed = processing_dataset()

    try:
        df = pd.read_json(dataset_processed)
        train_df, test_df = train_test_split(df, test_size=0.2)
        return {
            train_df.to_json(orient='records'),
            test_df.to_json(orient='records')
        }
    except FileNotFoundError:
        return {"error": "Dataset file not found."}

def train_dataset():
    train, test = split_dataset()

    train_df = pd.read_json(train)

    # Separating X_train and y_train
    X_train = train_df.drop(columns=["Species"])
    y_train = train_df["Species"]

    # Load model parameters from JSON file
    parameters_file_path = "services/epf-flower-data-science/src/config/model_parameters.json"
    with open(parameters_file_path, 'r') as file:
        model_parameters = json.load(file)

    # Initialize and train the model with train data
    model = RandomForestClassifier(**model_parameters)
    model.fit(X_train, y_train)

    if not os.path.exists('services/epf-flower-data-science/src/models'):
        os.makedirs('services/epf-flower-data-science/src/models')

    # Store the model
    model_save_path = 'services/epf-flower-data-science/src/models/random_forest_model.joblib'
    joblib.dump(model, model_save_path)

    return {"status": "Model trained and saved successfully"}

def predict():
    
    # Load the trained model
    model_save_path = 'services/epf-flower-data-science/src/models/random_forest_model.joblib'
    try:
        model = joblib.load(model_save_path)
    except FileNotFoundError:
        return {"error": "Trained model not found."}
    
    train, test = split_dataset()
    test_df = pd.read_json(test)
    
    X_test = test_df.drop(columns=["Species"])
    y_pred = pd.DataFrame(model.predict(X_test))

    return y_pred.to_json(orient="records")





    

