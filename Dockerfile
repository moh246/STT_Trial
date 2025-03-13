# Use the official RunPod PyTorch image as the base
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Set the shell to handle errors properly
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY STT_Requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project files
COPY . /app/

# Set environment variables required for RunPod execution
ENV PYTHONPATH="/app"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# RunPod expects a handler file, so rename or create one
COPY STT\ \(1\).py /app/handler.py  # Renaming your script to "handler.py"

# Set the command to run your handler file
CMD ["python", "handler.py"]
