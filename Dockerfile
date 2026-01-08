
# Use an official PyTorch base image with CUDA support (Updated to 2.4.0 for Llama 3.1 support)
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Set the working directory
WORKDIR /workspace/Neural-Pulse

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Upgrade pip and install requirements
# NOTE: We remove torch from requirements.txt to avoid overwriting the base image's optimized version
RUN pip install --upgrade pip && \
    grep -v "torch" requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# Copy source code
COPY . .

# Default command (can be overridden by K8s job)
CMD ["python", "datasets/oracle_validator.py", "--help"]
