# PDF Extractor API

API robusta para extração de informações de PDFs de contracheques utilizando processamento de imagem, GPU, IA e visão computacional avançada.

## Requisitos

- Docker
- Docker Compose
- NVIDIA Container Toolkit
- NVIDIA GPU com drivers CUDA
- Ollama (com modelo llama2)
- Conta no Replicate (para acesso ao Llava)

## Instalação com Docker (Recomendado)

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd <seu-repositorio>
```


3. Execute o script de instalação:
```bash
chmod +x install.sh
./install.sh
```

O script irá:
- Instalar Docker e Docker Compose (se necessário)
- Instalar NVIDIA Container Toolkit
- Construir e iniciar o container
- Configurar para iniciar automaticamente com o sistema

## Instalação Manual

Se preferir instalar manualmente:

1. Instale as dependências do sistema:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils libmagic1

# macOS
brew install tesseract tesseract-lang poppler libmagic
```

2. Instale o CUDA Toolkit (se ainda não tiver):
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

3. Instale as dependências Python:
```bash
sudo apt install python3.12-venv
sudo apt install python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Baixe o modelo spaCy para português:
```bash
python -m spacy download pt_core_news_lg
```

5. Configure o token do Replicate:
```bash
export REPLICATE_API_TOKEN="seu_token_aqui"
```

6. Certifique-se que o Ollama está rodando e acessível em http://192.168.15.222:11434

## Uso

### Com Docker

O serviço estará disponível automaticamente em http://localhost:8000 após a instalação.

Para gerenciar o serviço:
```bash
# Verificar status
sudo systemctl status pdf-extractor

# Reiniciar serviço
sudo systemctl restart pdf-extractor

# Ver logs
docker-compose logs -f
```

### Sem Docker

1. Inicie o servidor:
```bash
python main.py
```

2. A API estará disponível em http://localhost:8000

## Endpoint de Extração

Use o endpoint `/extrair` para processar PDFs:
```bash
curl -X POST "http://localhost:8000/extrair" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "pdf_file=@seu_arquivo.pdf"
```

## Estrutura da Resposta

A API retorna um JSON com a seguinte estrutura:

```json
{
    "nome_completo": "string",
    "matricula": "string",
    "mes_ano_referencia": "string",
    "vantagens": [
        {
            "codigo": "string",
            "descricao": "string",
            "percentual_duracao": "string",
            "valor": 0.0
        }
    ]
}
```

## Vantagens Processadas

- VENCIMENTO
- GRAT.A.FIS
- GRAT.A.FIS JUD
- AD.T.SERV
- CET-H.ESP
- PDF
- AD.NOT.INCORP
- DIF SALARIO/RRA
- TOTAL INFORMADO

## Recursos Avançados

- Processamento em GPU com CUDA
- OCR otimizado com EasyOCR
- Processamento de linguagem natural com spaCy
- Reconhecimento de entidades nomeadas com BERT
- Extração de texto de PDFs digitais e escaneados
- Pré-processamento de imagem avançado
- Matching fuzzy para identificação de vantagens
- Fallback para Ollama quando necessário
- **Análise visual com Llava-13b para maior precisão**
- **Detecção inteligente de percentuais e valores**
- **Remoção de duplicatas e consolidação de dados**
- **Validação cruzada de informações**

## Precisão e Confiabilidade

O sistema utiliza múltiplas estratégias para garantir a máxima precisão:

1. **Análise Visual com Llava-13b**: 
   - Modelo de visão computacional avançado
   - Capaz de entender o contexto visual do contracheque
   - Diferencia com precisão entre percentuais e valores

2. **Validação Cruzada**:
   - Compara resultados de diferentes métodos de extração
   - Usa posicionamento relativo para identificar valores corretos
   - Verifica consistência entre percentuais e valores

3. **Processamento Inteligente**:
   - Detecta e corrige confusões entre percentuais e valores
   - Consolida informações de múltiplas extrações
   - Remove duplicatas mantendo as informações mais completas

## Documentação da API

A documentação interativa da API está disponível em:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 