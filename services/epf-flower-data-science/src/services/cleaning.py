from sklearn.preprocessing import StandardScaler
import pandas as pd

#processing = remove the iris- in the species column
def processing():
    file_path = 'services/epf-flower-data-science/src/data/Iris.csv'
    try:
        df = pd.read_csv(file_path)
        df['Species'] = df['Species'].apply(lambda x: x.replace('Iris-', ''))  
        return df.to_json(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}