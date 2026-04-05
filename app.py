from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from pathlib import Path

app = FastAPI()
model_path = Path("data/minifrida/").absolute()

tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModel.from_pretrained(model_name)

class EmbRequest(BaseModel):
    text : str

@app.post("/embed")
def embed(req : Embrequest):
    inputs = tokenizer(req.text, return_tenzors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squezze().tolist()
    return {"embedding" : embeding}