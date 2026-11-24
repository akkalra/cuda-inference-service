FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-base.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
EXPOSE 8000
CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
