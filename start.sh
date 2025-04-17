#!/bin/bash

# Criar diretórios de cache se não existirem
mkdir -p /app/cache/transformers
mkdir -p /app/cache/torch
mkdir -p /app/cache/huggingface
mkdir -p /app/cache/easyocr

# Configurar permissões
chmod -R 777 /app/cache

# Iniciar a aplicação
echo "Iniciando a aplicação..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload 