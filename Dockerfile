FROM python:3.11-slim

WORKDIR /app

COPY ./api /app/api

COPY ./certs /app/certs

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
