from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
import json
import requests
from pdf_processor import PDFProcessor

load_dotenv()

app = FastAPI(title="PDF Extractor API",
             description="API para extração de informações de PDFs de contracheques",
             version="1.0.0")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Vantagem(BaseModel):
    codigo: str
    descricao: str
    percentual_duracao: Optional[str]
    valor: float

class ExtracaoResponse(BaseModel):
    nome_completo: str
    matricula: str
    mes_ano_referencia: str
    vantagens: List[Vantagem]
    total_informado: float
    total_calculado: float
    confere: bool

@app.post("/extrair", response_model=ExtracaoResponse)
async def extrair_informacoes(pdf_file: UploadFile = File(...)):
    try:
        # Salvar o arquivo temporariamente
        temp_path = f"temp_{pdf_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await pdf_file.read()
            buffer.write(content)
        
        # Processar o PDF
        processor = PDFProcessor()
        resultado = processor.process_pdf(temp_path)
        
        # Limpar arquivo temporário
        os.remove(temp_path)
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 