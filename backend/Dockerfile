FROM python:3.10-slim

WORKDIR /app

# Install only the necessary build dependencies for pymssql
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        freetds-dev \
        freetds-bin \
        build-essential \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]