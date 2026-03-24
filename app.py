from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_next_words

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    words = predict_next_words(data.text)
    return {"suggestions": words}