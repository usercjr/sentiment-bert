from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="IMDb Sentiment API", version="1.0.0")

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label = "positive" if predicted_class == 1 else "negative"
    return {"label": label}

