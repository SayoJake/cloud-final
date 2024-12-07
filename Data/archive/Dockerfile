# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and application code
COPY requirements.txt /app/
COPY app.py /app/
COPY templates /app/templates

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on (Flask default is 5000)
EXPOSE 8080

# Set the command to run the app
CMD ["python", "app.py"]
