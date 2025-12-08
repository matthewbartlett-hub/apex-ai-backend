# Use lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libmagic1

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start FastAPI app using uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8080"]
