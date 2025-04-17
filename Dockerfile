FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libmagic1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Cria e ativa ambiente virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia apenas os arquivos de dependências primeiro
COPY requirements.txt .

# Instala as dependências em camadas para melhor cache
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Instala outras dependências pesadas que raramente mudam
RUN pip install \
    transformers==4.35.0 \
    easyocr==1.7.1 \
    spacy==3.7.2

# Instala o modelo do spaCy
RUN python -m spacy download pt_core_news_lg

# Instala o restante das dependências
RUN pip install --no-cache-dir -r requirements.txt

# Cria diretório para cache
RUN mkdir -p /app/cache

# Copia os arquivos de código
COPY main.py .
COPY pdf_processor.py .
COPY start.sh /start.sh

# Configura permissões
RUN chmod +x /start.sh && \
    chmod -R 777 /app/cache

# Volume para cache persistente
VOLUME ["/app/cache"]

EXPOSE 8000

CMD ["/start.sh"]
