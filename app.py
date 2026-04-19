from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import time
import logging
from pathlib import Path

app = FastAPI()
model_path = "sergeyzh/rubert-mini-frida"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

class EmbRequest(BaseModel):
    text : str

@app.post("/embed")
def embed(req : EmbRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    time_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    time_end = time.perf_counter()

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return {"embedding" : embedding, "inference_time" : time_end - time_start}