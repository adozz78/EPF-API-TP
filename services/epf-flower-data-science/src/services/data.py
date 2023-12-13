from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
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
        # convert the df to a list
        return df.to_json(orient='records')  
    except FileNotFoundError:
        return {"error : file not found."}
    

