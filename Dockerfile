FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Evitar interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    libgl1-mesa-glx \
    python3-pip \
    python3.10-venv \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório da aplicação
WORKDIR /app

# Copiar arquivos do projeto
COPY requirements.txt .
COPY main.py .
COPY pdf_processor.py .

# Criar e ativar ambiente virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Baixar modelo spaCy
RUN python -m spacy download pt_core_news_lg

# Configurar variáveis de ambiente para Replicate
ENV REPLICATE_API_TOKEN="seu_token_aqui"

# Expor porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 