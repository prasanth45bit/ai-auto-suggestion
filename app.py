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

@app.get("/")
def home():
    return "Hello"

if __name__ == "__main__": 
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    