FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# EXPOSE is optional on Render; can leave or remove
EXPOSE 8000

# Use Render's PORT environment variable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
