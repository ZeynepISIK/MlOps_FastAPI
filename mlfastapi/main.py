from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class modelShema(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

@app.get("/")
def hello():
    return {"hello": "hello world"}

@app.post("/predict/knn/")
def predict_model(predict_value:modelShema):
    load_model = pickle.load(open("finalized_model.sav","rb"))
    df = pd.DataFrame(
        [predict_value.dict().values()
            ],
        columns=predict_value.dict().keys())

    predict = load_model.predict(df)
    return {"predict":int(predict[0])}
