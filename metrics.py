import requests
import time
import numpy as np
import psutil as ps
import os

N_REQUESTS = 2000

text = "This text is example. We are trying to test a system. We test embedding taking from rupert-mini-frida."
len_of_text = np.array(text.split()).shape[0]
address = "http://localhost:8000/embed"

latency = []
inference_time = []

for i in range(N_REQUESTS):
    time_start = time.perf_counter()
    req = requests.post(address, json={"text": text})
    data = req.json()
    inference_time.append(data["inference_time"])
    time_end = time.perf_counter()
    if req.status_code != 200:
        continue
    latency.append(time_end - time_start)

time_start = time.perf_counter()
for i in range(N_REQUESTS):
    req = requests.post(address, json={"text": text})
    data = req.json()
    if req.status_code != 200:
        raise Exception(f"Process finished with code {req.status_code}!")
time_end = time.perf_counter()
total_time = time_end - time_start
wps = N_REQUESTS * len_of_text / total_time
rps = N_REQUESTS / total_time

latency = np.sort(latency)
p50 = np.percentile(latency, 50)
p95 = np.percentile(latency, 95)
p99 = np.percentile(latency, 99)
mean_value = np.mean(latency)
mean_inference_time = np.mean(inference_time)
median_inference_time = np.percentile(inference_time, 50)
print(f"P50 = {p50 * 1000:.2f} мс")
print(f"P95 = {p95 * 1000:.2f} мс")
print(f"P99 = {p99 * 1000:.2f} мс")
print(f"Mean time = {mean_value * 1000:.2f} мс")
print(f"Mean inference time = {mean_inference_time * 1000:.2f} мс")
print(f"Median inference time = {median_inference_time * 1000:.2f} мс")
print(f"Total time = {total_time * 1000:.2f} мс")
print(f"RPS = {rps:.2f}")
print(f"WPS = {wps:.2f}")