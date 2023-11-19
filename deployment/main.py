from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os

app = FastAPI()

class InputData(BaseModel):
    minimum_nights: int
    reviews_per_month: float
    number_of_reviews_ltm: int
    calculated_host_listings_count: int
    license_Exempt: int
    neighbourhood_group_Eixample: int
    room_type_Entire_home_apt: int = Field(..., alias='room_type_Entire home/apt')
    room_type_Private_room: int = Field(..., alias='room_type_Private room')

# Load model from MLFlow
model = mlflow.pyfunc.load_model("models:/{model_name}/{model_tier}".format(model_name=os.environ["MODEL_NAME"], model_tier=os.environ["MODEL_NUMBER"]))

@app.post("/predict")
def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict(by_alias=True)])

    # Convert 'neighbourhood_group_Eixample' to int32
    input_df['neighbourhood_group_Eixample'] = input_df['neighbourhood_group_Eixample'].astype('int32')
    input_df['room_type_Entire home/apt'] = input_df['room_type_Entire home/apt'].astype('int32')
    input_df['room_type_Private room'] = input_df['room_type_Private room'].astype('int32')

    prediction = np.exp(model.predict(input_df))
    prediction_list = prediction.tolist()
    return {"prediction": prediction_list}

# Healthcheck endpoint needed for app runner service
@app.get("/health")
def health():
     return {"Healthy": True}
