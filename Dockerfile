
# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from creating cache files
ENV PYTHONDONTWRITEBYTECODE=1

# Display logs immediately
ENV PYTHONUNBUFFERED=1

# Install application dependencies
COPY requirements-serve.txt .

RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application code
COPY app.py .

# Copy trained model
COPY model/ ./model/

# Copy required fare dataset
COPY data/domestic_fares_clean.csv ./data/

# Copy complete frontend
COPY static/ ./static/

# Create logs directory
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Start FlySmart application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
