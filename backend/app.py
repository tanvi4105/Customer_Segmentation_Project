# from fastapi import FastAPI, UploadFile, File
# import pandas as pd
# from predictor import predict_customer
# from schemas import CustomerInput

# app = FastAPI(title="Customer Segmentation API")

# profile = pd.read_csv("../outputs/segmented_customers.csv")

# @app.get("/")
# def home():
#     return {"message": "Customer Segmentation API Running"}

# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# @app.post("/segment")
# def segment(customer: CustomerInput):
#     return predict_customer(customer.dict())

# @app.post("/batch-segment")
# async def batch_segment(file: UploadFile = File(...)):

#     df = pd.read_csv(file.file)

#     predictions = []

#     for _, row in df.iterrows():
#         pred = predict_customer(row.to_dict())
#         predictions.append(pred)

#     result = pd.concat([df, pd.DataFrame(predictions)], axis=1)

#     return result.to_dict(orient="records")

# @app.get("/cluster-info")
# def cluster_info():

#     summary = profile.groupby("Segment_Name").agg({
#         "CustomerID":"count",
#         "Monetary":"mean",
#         "Frequency":"mean",
#         "Engagement":"mean",
#         "Recency":"mean"
#     }).round(2)

#     return summary.reset_index().to_dict(orient="records")

from fastapi import FastAPI, UploadFile, File
import pandas as pd
from pathlib import Path

from predictor import predict_customer
from schemas import CustomerInput

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------- App ----------------
app = FastAPI(title="Customer Segmentation API")

# Load segmented customer data
profile = pd.read_csv(OUTPUT_DIR / "segmented_customers.csv")


# ---------------- Home ----------------
@app.get("/")
def home():
    return {"message": "Customer Segmentation API Running"}


# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- Single Prediction ----------------
@app.post("/segment")
def segment(customer: CustomerInput):
    return predict_customer(customer.model_dump())


# ---------------- Batch Prediction ----------------
@app.post("/batch-segment")
async def batch_segment(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    predictions = []

    for _, row in df.iterrows():
        predictions.append(predict_customer(row.to_dict()))

    result = pd.concat([df, pd.DataFrame(predictions)], axis=1)

    return result.to_dict(orient="records")


# ---------------- Cluster Information ----------------
@app.get("/cluster-info")
def cluster_info():

    summary = profile.groupby("Segment_Name").agg({
        "CustomerID": "count",
        "Monetary": "mean",
        "Frequency": "mean",
        "Recency": "mean",
        "AvgOrderValue": "mean",
        "TotalQuantity": "mean"
    }).round(2)

    summary = summary.rename(columns={
        "CustomerID": "CustomerCount",
        "Monetary": "AverageSpend",
        "Frequency": "AverageOrders",
        "Recency": "AverageRecency",
        "AvgOrderValue": "AverageOrderValue",
        "TotalQuantity": "AverageItems"
    })

    return summary.reset_index().to_dict(orient="records")