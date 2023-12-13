from fastapi import APIRouter, HTTPException
from src.services.data import download_iris_dataset,load_iris_dataset
from src.services.cleaning import processing

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
    processed_dataset = processing()
    if "error" in processed_dataset:
        raise HTTPException(status_code=404, detail=processed_dataset["error"])
    return processed_dataset