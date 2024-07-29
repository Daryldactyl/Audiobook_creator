# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port streamlit runs on
EXPOSE 8501

# Run streamlit app when the container launches
CMD ["streamlit", "run", "--server.enableCORS", "false", "--server.port", "8501", "audio_book.py"]
