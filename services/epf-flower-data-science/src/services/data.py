from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

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
            "train": train_df.to_json(orient='records'),
            "test": test_df.to_json(orient='records')
        }
    except FileNotFoundError:
        return {"error": "Dataset file not found."}

    

