FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pdf2image \
    pillow \
    opencv-python \
    numpy \
    torch \
    torchvision \
    transformers \
    "transformers[torch]" \
    sentencepiece \
    protobuf \
    python-dotenv \
    && pip install --no-cache-dir easyocr

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos da aplicação
COPY main.py .
COPY pdf_processor.py .

# Expor porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
