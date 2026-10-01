
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------- Load Models ----------------
kmeans = joblib.load(MODEL_DIR / "clustering_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

# ---------------- Load Cluster Mapping ----------------
segmented_df = pd.read_csv(OUTPUT_DIR / "segmented_customers.csv")

# Automatically create cluster → segment mapping
segment_names = (
    segmented_df[["Cluster", "Segment_Name"]]
    .drop_duplicates()
    .set_index("Cluster")["Segment_Name"]
    .to_dict()
)

# ---------------- Business Recommendations ----------------
recommendations = {
    "High Value Loyal": "Offer VIP rewards, exclusive products, and premium membership.",
    "Frequent Buyer": "Recommend complementary products and loyalty rewards.",
    "Occasional Customer": "Send personalized emails and re-engagement campaigns.",
    "At Risk": "Provide special discounts and win-back offers."
}

# ---------------- Prediction Function ----------------
def predict_customer(data):

    df = pd.DataFrame([data])

    # Keep feature order exactly as used during training
    X = scaler.transform(df[feature_columns])

    # PCA transformation
    X_pca = pca.transform(X)

    # Predict cluster
    cluster = int(kmeans.predict(X_pca)[0])

    # Distance from nearest centroid
    distance = float(np.min(kmeans.transform(X_pca)))

    # Get segment name dynamically
    segment = segment_names.get(cluster, "Unknown")

    return {
        "cluster": cluster,
        "segment_name": segment,
        "distance_from_centroid": round(distance, 2),
        "recommendation": recommendations.get(segment, "No recommendation available.")
    }
