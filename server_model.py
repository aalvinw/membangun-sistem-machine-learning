from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="API Prediksi Pengiriman")

# Load model
model = joblib.load("model.pkl")

# Skema input berdasarkan preprocessed_data.csv
class DeliveryFeatures(BaseModel):
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: float
    Prior_purchases: int
    Discount_offered: float
    Weight_in_gms: float
    Warehouse_block_A: int
    Warehouse_block_B: int
    Warehouse_block_C: int
    Warehouse_block_D: int
    Warehouse_block_F: int
    Mode_of_Shipment_Flight: int
    Mode_of_Shipment_Road: int
    Mode_of_Shipment_Ship: int
    Product_importance_high: int
    Product_importance_low: int
    Product_importance_medium: int
    Gender_F: int
    Gender_M: int


@app.post("/predict")
def predict(features: DeliveryFeatures):
    try:
        # Ubah input menjadi dataframe
        data = pd.DataFrame([features.dict()])
        # Prediksi
        prediction = model.predict(data)[0]
        status = "Tepat Waktu" if prediction == 1 else "Terlambat"
        return {"prediction": int(prediction), "status": status}
    except Exception as e:
        return {"error": str(e)}
