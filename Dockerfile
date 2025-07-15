# Use official Python image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Copy all necessary project folders and files needed at runtime
COPY app/ ./app/
COPY common/ ./common/
COPY domains/ ./domains/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY outputs/ ./outputs/ 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]