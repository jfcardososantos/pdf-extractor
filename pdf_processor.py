import os
import cv2
import numpy as np
import torch
import easyocr
import pdfplumber
import spacy
from pdf2image import convert_from_path
import pytesseract
from typing import Dict, List, Optional, Tuple
import json
import requests
from pydantic import BaseModel
from Levenshtein import ratio
import re
from transformers import pipeline

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

class PDFProcessor:
    def __init__(self):
        self.ollama_url = "http://192.168.15.222:11434/api/generate"
        self.vantagens_chave = [
            "VENCIMENTO", "GRAT.A.FIS", "GRAT.A.FIS JUD", "AD.T.SERV",
            "CET-H.ESP", "PDF", "AD.NOT.INCORP", "DIF SALARIO/RRA",
            "TOTAL INFORMADO"
        ]
        
        # Verificar se CUDA está disponível
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Inicializar EasyOCR com GPU
        self.reader = easyocr.Reader(['pt'], gpu=True)
        
        # Carregar modelo spaCy para português
        self.nlp = spacy.load("pt_core_news_lg")
        
        # Inicializar pipeline de NER
        self.ner_pipeline = pipeline("ner", model="neuralmind/bert-base-portuguese-cased", device=0 if torch.cuda.is_available() else -1)
        
        # Padrões de regex para extração
        self.patterns = {
            'matricula': r'MATR[ÍI]CULA[:\s]*(\d+)',
            'mes_ano': r'REFER[ÊE]NCIA[:\s]*([A-Za-z]+/\d{4})',
            'valor': r'R?\$?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)',
        }

    def process_pdf(self, pdf_path: str) -> ExtracaoResponse:
        # Tentar primeiro com pdfplumber para PDFs digitais
        texto_digital = self._extrair_texto_digital(pdf_path)
        
        # Se não conseguir extrair texto suficiente, converter para imagem
        if len(texto_digital.strip()) < 100:
            texto_completo = self._processar_como_imagem(pdf_path)
        else:
            texto_completo = texto_digital
        
        # Usar múltiplas estratégias para extrair informações
        resultado = self._extrair_informacoes_estruturadas(texto_completo)
        
        return resultado

    def _extrair_texto_digital(self, pdf_path: str) -> str:
        texto = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    texto += page.extract_text() or ""
        except Exception as e:
            print(f"Erro ao extrair texto digital: {str(e)}")
        return texto

    def _processar_como_imagem(self, pdf_path: str) -> str:
        # Converter PDF para imagens
        images = convert_from_path(pdf_path)
        
        texto_completo = ""
        for image in images:
            # Converter para formato OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Pré-processamento da imagem
            processed_image = self._preprocess_image(opencv_image)
            
            # Extrair texto usando EasyOCR (GPU)
            resultados = self.reader.readtext(processed_image)
            texto = " ".join([result[1] for result in resultados])
            
            # Backup com Tesseract se necessário
            if len(texto.strip()) < 50:
                texto = pytesseract.image_to_string(processed_image, lang='por')
            
            texto_completo += texto + "\n"
        
        return texto_completo

    def _preprocess_image(self, image):
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Remover ruído
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Aumentar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def _extrair_informacoes_estruturadas(self, texto: str) -> ExtracaoResponse:
        # Extrair informações usando regex
        matricula = self._extrair_com_regex(texto, self.patterns['matricula'])
        mes_ano = self._extrair_com_regex(texto, self.patterns['mes_ano'])
        
        # Processar texto com spaCy
        doc = self.nlp(texto)
        
        # Extrair nome usando NER
        nome_completo = self._extrair_nome(doc)
        
        # Extrair vantagens
        vantagens = self._extrair_vantagens(texto)
        
        # Validar e normalizar dados
        if not nome_completo or not matricula or not mes_ano:
            # Usar Ollama como fallback
            return self._usar_ollama_fallback(texto)
        
        return ExtracaoResponse(
            nome_completo=nome_completo,
            matricula=matricula,
            mes_ano_referencia=mes_ano,
            vantagens=vantagens
        )

    def _extrair_com_regex(self, texto: str, pattern: str) -> str:
        match = re.search(pattern, texto, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extrair_nome(self, doc) -> str:
        # Procurar por entidades de pessoa
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        
        # Fallback: procurar por padrões comuns de nome
        nome_pattern = r"NOME[:\s]*([A-Za-zÀ-ÿ\s]+)"
        match = re.search(nome_pattern, doc.text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extrair_vantagens(self, texto: str) -> List[Vantagem]:
        vantagens = []
        linhas = texto.split('\n')
        
        for linha in linhas:
            for vantagem in self.vantagens_chave:
                if ratio(vantagem.lower(), linha.lower()) > 0.8:  # Usar Levenshtein para matching fuzzy
                    # Extrair valor
                    valor_match = re.search(self.patterns['valor'], linha)
                    valor = self._normalizar_valor(valor_match.group(1)) if valor_match else 0.0
                    
                    # Extrair percentual/duração se existir
                    percentual = ""
                    if "%" in linha:
                        percentual_match = re.search(r'(\d+(?:,\d+)?%)', linha)
                        if percentual_match:
                            percentual = percentual_match.group(1)
                    
                    vantagens.append(Vantagem(
                        codigo=vantagem,
                        descricao=vantagem,
                        percentual_duracao=percentual,
                        valor=valor
                    ))
        
        return vantagens

    def _usar_ollama_fallback(self, texto: str) -> ExtracaoResponse:
        prompt = f"""
        Analise o seguinte texto de um contracheque e extraia as informações solicitadas.
        Retorne apenas um JSON válido com a seguinte estrutura:
        {{
            "nome_completo": "nome do funcionário",
            "matricula": "número da matrícula",
            "mes_ano_referencia": "mês/ano de referência",
            "vantagens": [
                {{
                    "codigo": "código da vantagem",
                    "descricao": "descrição da vantagem",
                    "percentual_duracao": "percentual ou duração (se houver)",
                    "valor": valor numérico
                }}
            ]
        }}

        Texto do contracheque:
        {texto}
        """

        response = requests.post(
            self.ollama_url,
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception("Erro ao processar texto com Ollama")

        try:
            resultado_texto = response.json()["response"]
            inicio_json = resultado_texto.find("{")
            fim_json = resultado_texto.rfind("}") + 1
            json_str = resultado_texto[inicio_json:fim_json]
            
            dados = json.loads(json_str)
            return ExtracaoResponse(**dados)
            
        except Exception as e:
            raise Exception(f"Erro ao processar resposta do Ollama: {str(e)}")

    def _normalizar_valor(self, valor_str: str) -> float:
        try:
            valor_str = valor_str.replace("R$", "").strip()
            valor_str = valor_str.replace(".", "").replace(",", ".")
            return float(valor_str)
        except:
            return 0.0 