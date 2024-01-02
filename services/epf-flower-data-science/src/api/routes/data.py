from fastapi import APIRouter, HTTPException
from src.services.data import download_iris_dataset,load_iris_dataset,processing_dataset,split_dataset, train_dataset, predict, get_firestore_data, update_firestore_data

router = APIRouter()

@router.get("/dowload-data")
def download_data():
    return download_iris_dataset()

@router.get("/load-data")
def load_data():
    dataset = load_iris_dataset()
    if "error" in dataset:
        raise HTTPException(status_code=404, detail=dataset["error"])
    return dataset

@router.get("/processing")
def get_processed_iris_dataset():
    processed_dataset = processing_dataset()
    if "error" in processed_dataset:
        raise HTTPException(status_code=404, detail=processed_dataset["error"])
    return processed_dataset

@router.get("/split")
def split_iris_dataset():
    result = split_dataset()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/train")
def train_iris_dataset():
    result = train_dataset()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/predict")
def predict_iris_dataset():
    result = predict()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/get_firestore")
def get_firestore():
    result = get_firestore_data()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/update_firestore")
def update_firestore():
    result = update_firestore_data()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result