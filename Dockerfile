FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

RUN useradd --no-create-home appuser
USER appuser

ENV MODELS_DIR=models
ENV FRAUD_THRESHOLD=0.856
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
