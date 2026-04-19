FROM python:3.13

ENV PROJ_DIR=. LOG_LEVEL=debug
LABEL author="Roman Tsvetkov" version="0.0.0.1"

WORKDIR /app
COPY requirements.txt $PROJ_DIR


RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModel; n='sergeyzh/rubert-mini-frida'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]