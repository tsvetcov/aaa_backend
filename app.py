from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import time
import os
import psutil as ps
import logging
from pathlib import Path

app = FastAPI()
model_path = "sergeyzh/rubert-mini-frida"

cur_process = ps.Process(os.getpid())
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

class EmbRequest(BaseModel):
    text : str

@app.post("/embed")
def embed(req : EmbRequest):
    memory_before = cur_process.memory_info().rss / 1024 /1024
    cur_process.cpu_percent(None)
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    time_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    time_end = time.perf_counter()
    memory_after = cur_process.memory_info().rss / 1024 /1024
    cpu_usage = cur_process.cpu_percent(None)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return {"embedding" : embedding, "inference_time" : time_end - time_start, "memory" : memory_after, "memory_change" :  memory_after - memory_before, "cpu_usage" : cpu_usage}