import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import load_model, inference

# Initialize FastAPI app
app = FastAPI()

# Get the project path for loading artifacts
project_path = os.path.dirname(__file__)


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(...,
                                example="United-States",
                                alias="native-country")


# Load model artifacts
encoder = load_model(os.path.join(project_path, "model", "encoder.pkl"))
model = load_model(os.path.join(project_path, "model", "model.pkl"))
lb = load_model(os.path.join(project_path, "model", "lb.pkl"))


@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/data/")
async def post_inference(data: Data):
    # Convert the Pydantic model into a dict.
    data_dict = data.dict()

    # Clean up the dict to turn it into a Pandas DataFrame with proper hyphens
    data_raw = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data_raw)

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    # Process the data for inference
    data_processed, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run prediction
    _inference = inference(model, data_processed)

    # Convert prediction back to label (e.g., <=50K or >50K)
    prediction_label = lb.inverse_transform(_inference)[0]

    return {"result": str(prediction_label)}
