from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_next_words
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class InputText(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: InputText):
    words = predict_next_words(data.text)
    return {"suggestions": words}


@app.get("/")
def health():
    return {"status": "running"}

    
if __name__ == "__main__": 
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    