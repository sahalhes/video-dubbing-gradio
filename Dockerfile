# Use a lightweight Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Set ffmpeg permissions
RUN chmod +x /usr/bin/ffmpeg

# Expose the port for Gradio
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
