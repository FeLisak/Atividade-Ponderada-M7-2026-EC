FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py backend.py graficos.py ./
COPY data ./data

EXPOSE 8000

CMD ["python", "train.py"]
