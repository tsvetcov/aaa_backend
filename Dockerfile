FROM python:3.13

ENV FLASK_APP=app.py PROJ_DIR=. LOG_LEVEL=debug
LABEL author="Roman Tsvetkov" version="0.0.0.1"

WORKDIR /app
COPY requirements.txt $PROJ_DIR


RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

CMD ["unicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]