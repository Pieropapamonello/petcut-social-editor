FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data/exports
ENV PYTHONUNBUFFERED=1 PORT=10000
EXPOSE 10000
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--worker-class", "gthread", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app"]
