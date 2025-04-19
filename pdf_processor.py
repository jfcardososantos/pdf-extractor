import os
import cv2
import numpy as np
import torch
import easyocr
import pdfplumber
import spacy
from pdf2image import convert_from_path
import pytesseract
from typing import Dict, List, Optional, Tuple, Any
import json
import requests
from pydantic import BaseModel
from Levenshtein import ratio
import re
from transformers import pipeline
from PIL import Image
import io
import magic
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    total_vantagens: float

class PDFProcessor:
    def __init__(self):
        self.ollama_url = "http://192.168.15.222:11434/api/generate"
        self.vantagens_chave = [
            "VENCIMENTO", "GRAT.A.FIS", "GRAT.A.FIS JUD", "AD.T.SERV",
            "CET-H.ESP", "PDF", "AD.NOT.INCORP", "DIF SALARIO/RRA"
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
            'percentual': r'(\d+(?:,\d+)?%)',
            'total_vantagens': r'TOTAL\s+(?:DE\s+)?VANTAGENS[:\s]*R?\$?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)',
        }
        
        # Vetorizador para comparação de texto
        self.vectorizer = TfidfVectorizer()
        
        # Treinar vetorizador com as vantagens
        self.vectorizer.fit_transform(self.vantagens_chave)

    def process_pdf(self, pdf_path: str) -> ExtracaoResponse:
        # Verificar tipo de arquivo
        file_type = magic.from_file(pdf_path, mime=True)
        
        # Tentar primeiro com pdfplumber para PDFs digitais
        texto_digital = self._extrair_texto_digital(pdf_path)
        
        # Se não conseguir extrair texto suficiente, converter para imagem
        if len(texto_digital.strip()) < 100:
            texto_completo = self._processar_como_imagem(pdf_path)
        else:
            texto_completo = texto_digital
        
        # Usar múltiplas estratégias para extrair informações
        resultado = self._extrair_informacoes_estruturadas(texto_completo, pdf_path)
        
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
        # Converter PDF para imagens com alta resolução
        images = convert_from_path(pdf_path, dpi=400)  # Aumentando ainda mais o DPI
        
        texto_completo = ""
        for image in images:
            # Converter para formato OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Pré-processamento da imagem
            processed_image = self._preprocess_image(opencv_image)
            
            # Extrair texto usando EasyOCR com configurações específicas
            resultados = self.reader.readtext(
                processed_image,
                paragraph=False,  # Desativando agrupamento para manter estrutura
                allowlist='0123456789,.-ABCDEFGHIJKLMNOPQRSTUVWXYZ/% ',  # Lista de caracteres permitidos
                batch_size=4,
                detail=1  # Retornar detalhes para processamento adicional
            )
            
            # Processar resultados mantendo a estrutura tabular
            linhas_texto = []
            for result in resultados:
                bbox = result[0]  # Coordenadas do texto
                texto = result[1]  # Texto detectado
                conf = result[2]  # Confiança da detecção
                
                # Filtrar resultados com baixa confiança
                if conf < 0.5:
                    continue
                
                # Organizar por posição vertical (y) para manter estrutura
                y_pos = (bbox[0][1] + bbox[2][1]) / 2
                linhas_texto.append((y_pos, texto))
            
            # Ordenar por posição vertical e juntar texto
            linhas_texto.sort(key=lambda x: x[0])
            texto = "\n".join(texto for _, texto in linhas_texto)
            
            # Backup com Tesseract se necessário
            if len(texto.strip()) < 50:
                # Configurar Tesseract para tabelas
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                texto = pytesseract.image_to_string(
                    processed_image,
                    lang='por',
                    config=custom_config
                )
            
            texto_completo += texto + "\n"
        
        return texto_completo

    def _preprocess_image(self, image):
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumentar resolução
        height, width = gray.shape
        gray = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # Normalizar contraste
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Aplicar threshold adaptativo com parâmetros ajustados
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # Tamanho do bloco aumentado
            10   # Constante ajustada
        )
        
        # Remover ruído mantendo linhas finas
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Melhorar definição de texto
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel)
        
        return sharpened

    def _extrair_informacoes_estruturadas(self, texto: str, pdf_path: str) -> ExtracaoResponse:
        # Extrair informações usando regex
        matricula = self._extrair_com_regex(texto, self.patterns['matricula'])
        mes_ano = self._extrair_com_regex(texto, self.patterns['mes_ano'])
        
        # Processar texto com spaCy
        doc = self.nlp(texto)
        
        # Extrair nome usando NER
        nome_completo = self._extrair_nome(doc)
        
        # Extrair vantagens com Llava para maior precisão
        vantagens = self._extrair_vantagens_com_llava(texto, pdf_path)
        
        # Extrair total informado no documento
        total_vantagens = self._extrair_total_vantagens(texto)
        
        # Validar e normalizar dados
        if not nome_completo or not matricula or not mes_ano:
            # Usar Ollama como fallback
            return self._usar_ollama_fallback(texto)
        
        return ExtracaoResponse(
            nome_completo=nome_completo,
            matricula=matricula,
            mes_ano_referencia=mes_ano,
            vantagens=vantagens,
            total_vantagens=total_vantagens,
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

    def _extrair_vantagens_com_llava(self, texto: str, pdf_path: str) -> List[Vantagem]:
        vantagens = []
        
        # Extrair vantagens usando Llava
        response = self._analisar_com_llava(pdf_path)
        if response and 'response' in response:
            for item in response['response']:
                try:
                    # Extrair campos com valores padrão
                    codigo = str(item.get('codigo', '')).strip()
                    descricao = str(item.get('descricao', '')).strip()
                    percentual = str(item.get('percentual', '')).strip()
                    valor_str = str(item.get('valor', '')).strip()
                    
                    # Validar campos obrigatórios
                    if not codigo or not descricao:
                        print(f"Item sem código ou descrição: {item}")
                        continue
                    
                    # Converter valor para float
                    valor = 0.0
                    try:
                        # Limpar o valor antes de converter
                        valor_str = valor_str.replace('R$', '').replace(' ', '').strip()
                        valor = float(valor_str.replace('.', '').replace(',', '.'))
                    except ValueError:
                        print(f"Erro ao converter valor: {valor_str}")
                        continue
                    
                    # Criar objeto Vantagem
                    vantagem = Vantagem(
                        codigo=codigo,
                        descricao=descricao,
                        percentual_duracao=percentual,
                        valor=valor
                    )
                    vantagens.append(vantagem)
                    print(f"Vantagem extraída: {vantagem}")
                except Exception as e:
                    print(f"Erro ao processar item: {item}")
                    print(f"Erro: {str(e)}")
                    continue
                    
        return vantagens

    def _analisar_com_llava(self, img_path: str) -> str:
        # Converter imagem para base64
        with open(img_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Preparar prompt para Llava
        prompt = """
        Analise esta imagem de um contracheque. Na seção VANTAGENS, existem as seguintes colunas:
        - Cód. (código da vantagem)
        - Descrição (nome da vantagem)
        - Perct./Horas (percentual ou horas)
        - Período (ignorar esta coluna)
        - Valor(R$) (valor monetário)

        Para cada linha da tabela de vantagens:
        1. Identifique o código na primeira coluna
        2. Identifique a descrição na segunda coluna
        3. Capture o percentual/horas na terceira coluna (se existir)
        4. Capture o valor na última coluna

        Retorne APENAS um JSON válido com a seguinte estrutura:
        {
            "response": [
                {
                    "codigo": "0002",
                    "descricao": "Vencimento",
                    "percentual": "30.00",
                    "valor": "3052.41"
                }
            ]
        }

        Importante:
        - Ignore completamente a coluna "Período"
        - Mantenha os zeros à esquerda nos códigos
        - Preserve os valores exatamente como aparecem
        - Retorne apenas o JSON, sem texto adicional
        """
        
        # Enviar para Gemma via Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "gemma3:12b",
                    "prompt": prompt,
                    "stream": False,
                    "images": [img_data]
                }
            )
            
            if response.status_code != 200:
                print(f"Erro ao usar Gemma3: {response.status_code}")
                return ""
            
            # Tentar extrair o JSON da resposta
            resposta = response.json()["response"]
            
            # Limpar a resposta para garantir que é um JSON válido
            resposta = resposta.strip()
            inicio_json = resposta.find("{")
            fim_json = resposta.rfind("}") + 1
            
            if inicio_json == -1 or fim_json == 0:
                print("Não foi possível encontrar um JSON válido na resposta")
                return ""
            
            json_str = resposta[inicio_json:fim_json]
            
            # Tentar fazer o parse do JSON
            try:
                dados = json.loads(json_str)
                return dados
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {str(e)}")
                print(f"JSON problemático: {json_str}")
                return ""
                
        except Exception as e:
            print(f"Erro ao usar Gemma3: {str(e)}")
            return ""

    def _processar_resposta_llava(self, resposta: str) -> List[Vantagem]:
        vantagens = []
        
        # Dividir resposta em linhas
        linhas = resposta.split('\n')
        
        for linha in linhas:
            # Procurar por vantagens conhecidas
            for vantagem in self.vantagens_chave:
                if ratio(vantagem.lower(), linha.lower()) > 0.8:
                    # Extrair valor
                    valor_match = re.search(self.patterns['valor'], linha)
                    valor = self._normalizar_valor(valor_match.group(1)) if valor_match else 0.0
                    
                    # Extrair percentual/duração
                    percentual = ""
                    percentual_match = re.search(self.patterns['percentual'], linha)
                    if percentual_match:
                        percentual = percentual_match.group(1)
                    
                    # Verificar se o valor não é na verdade um percentual
                    if valor > 0 and percentual:
                        # Verificar qual é mais provável ser o valor monetário
                        valor_pos = linha.find(str(valor))
                        percentual_pos = linha.find(percentual)
                        
                        # Se o percentual está antes do valor, pode ser que estejam trocados
                        if percentual_pos < valor_pos:
                            # Trocar valores
                            temp_valor = valor
                            valor = float(percentual.replace('%', '').replace(',', '.'))
                            percentual = f"{temp_valor}%"
                    
                    vantagens.append(Vantagem(
                        codigo=vantagem,
                        descricao=vantagem,
                        percentual_duracao=percentual,
                        valor=valor
                    ))
        
        return vantagens

    def _remover_duplicatas(self, vantagens: List[Vantagem]) -> List[Vantagem]:
        # Usar conjunto para remover duplicatas
        vantagens_dict = {}
        
        for v in vantagens:
            # Usar código como chave
            if v.codigo not in vantagens_dict:
                vantagens_dict[v.codigo] = v
            else:
                # Se já existe, verificar qual tem mais informações
                existente = vantagens_dict[v.codigo]
                
                # Se o novo tem percentual e o existente não, atualizar
                if v.percentual_duracao and not existente.percentual_duracao:
                    vantagens_dict[v.codigo] = v
                # Se o novo tem valor e o existente não, atualizar
                elif v.valor > 0 and existente.valor == 0:
                    vantagens_dict[v.codigo] = v
        
        return list(vantagens_dict.values())

    def _extrair_total_vantagens(self, texto: str) -> float:
        # Padrão específico para o total de vantagens no contracheque
        padrao = r'TOTAL\s+DE\s+VANTAGENS\s*[\r\n]*\s*([\d\.,]+)'
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            valor_str = match.group(1).strip()
            try:
                # Tratar diferentes formatos de número
                valor_str = valor_str.replace('.', '').replace(',', '.')
                return float(valor_str)
            except ValueError:
                print(f"Erro ao converter valor total: {valor_str}")
        return 0.0

    def _usar_ollama_fallback(self, texto: str) -> ExtracaoResponse:
        prompt = f"""
        Analise o seguinte texto de um contracheque e extraia as informações solicitadas.
        Retorne APENAS um JSON válido com a seguinte estrutura, sem nenhum texto adicional:
        {{
            "nome_completo": "nome do funcionário",
            "matricula": "número da matrícula",
            "mes_ano_referencia": "mês/ano de referência",
            "vantagens": [
                {{
                    "codigo": "código da vantagem",
                    "descricao": "descrição da vantagem",
                    "percentual_duracao": "",
                    "valor": 0.0
                }}
            ],
            "total_vantagens": 0.0
        }}

        Texto do contracheque:
        {texto}
        """

        response = requests.post(
            self.ollama_url,
            json={
                "model": "gemma3:12b",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception("Erro ao processar texto com Ollama")

        try:
            resultado_texto = response.json()["response"]
            
            # Limpar o texto para garantir um JSON válido
            resultado_texto = resultado_texto.strip()
            
            # Encontrar o primeiro '{' e o último '}'
            inicio_json = resultado_texto.find("{")
            fim_json = resultado_texto.rfind("}") + 1
            
            if inicio_json == -1 or fim_json == 0:
                raise Exception("Não foi possível encontrar um JSON válido na resposta")
            
            json_str = resultado_texto[inicio_json:fim_json]
            
            # Tentar limpar caracteres inválidos
            json_str = json_str.replace('\n', ' ').replace('\r', '')
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove vírgulas extras antes de }
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove vírgulas extras antes de ]
            
            # Tentar fazer o parse do JSON
            try:
                dados = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {str(e)}")
                print(f"JSON problemático: {json_str}")
                raise
            
            # Validar campos obrigatórios
            campos_obrigatorios = ["nome_completo", "matricula", "mes_ano_referencia", "vantagens"]
            for campo in campos_obrigatorios:
                if campo not in dados:
                    raise Exception(f"Campo obrigatório '{campo}' não encontrado na resposta")
            
            # Validar vantagens de forma mais flexível
            for v in dados["vantagens"]:
                # Garantir que todos os campos são strings
                v["codigo"] = str(v.get("codigo", "")).strip()
                v["descricao"] = str(v.get("descricao", "")).strip()
                v["percentual_duracao"] = str(v.get("percentual_duracao", "")).strip()
                
                # Se não tiver código, usa a descrição
                if not v["codigo"] and v["descricao"]:
                    v["codigo"] = v["descricao"]
                
                # Se não tiver descrição, usa o código
                if not v["descricao"] and v["codigo"]:
                    v["descricao"] = v["codigo"]
                
                # Garantir que o valor é um número
                try:
                    v["valor"] = float(str(v.get("valor", "0")).replace('R$', '').replace('.', '').replace(',', '.'))
                except ValueError:
                    v["valor"] = 0.0
            
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