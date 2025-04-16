# PDF Extractor API

API robusta para extração de informações de PDFs de contracheques utilizando processamento de imagem, GPU e IA.

## Requisitos

- Python 3.8+
- CUDA 11.8+ (para aproveitar a GPU)
- Tesseract OCR
- Poppler
- Ollama (com modelo llama2)

## Instalação

1. Instale as dependências do sistema:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils

# macOS
brew install tesseract tesseract-lang poppler
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

5. Certifique-se que o Ollama está rodando e acessível em http://192.168.15.222:11434

## Uso

1. Inicie o servidor:
```bash
python main.py
```

2. A API estará disponível em http://localhost:8000

3. Use o endpoint `/extrair` para processar PDFs:
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

## Documentação da API

A documentação interativa da API está disponível em:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 