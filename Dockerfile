# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (without GPU)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Set Render dynamic PORT
ENV PORT 10000

# Run your FastAPI app
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
