# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /mnist

# copy requirements and install them
COPY mnist/requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app code
COPY mnist/ .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Default cmd to run the inference API
CMD [ "python", "serve.py" ]


