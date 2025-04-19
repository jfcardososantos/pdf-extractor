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
import warnings
import logging

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
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Suprimir avisos específicos
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.conv')
        warnings.filterwarnings('ignore', message='Some weights of BertForTokenClassification')
        
        self.ollama_url = "http://192.168.15.222:11434/api/generate"
        self.vantagens_chave = [
            "VENCIMENTO", "GRAT.A.FIS", "GRAT.A.FIS JUD", "AD.T.SERV",
            "CET-H.ESP", "PDF", "AD.NOT.INCORP", "DIF SALARIO/RRA"
        ]
        
        # Inicializar dispositivo CUDA com tratamento de erro
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Usando dispositivo: {self.device}")
        except Exception as e:
            self.logger.warning(f"Erro ao configurar CUDA, usando CPU: {str(e)}")
            self.device = torch.device("cpu")
        
        # Inicializar EasyOCR com tratamento de erro
        try:
            self.reader = easyocr.Reader(['pt'], gpu=torch.cuda.is_available())
            self.logger.info("EasyOCR inicializado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar EasyOCR: {str(e)}")
            raise
        
        # Carregar modelo spaCy
        try:
            self.nlp = spacy.load("pt_core_news_lg")
            self.logger.info("Modelo spaCy carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo spaCy: {str(e)}")
            raise
        
        # Inicializar pipeline de NER com tratamento de erro
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="neuralmind/bert-base-portuguese-cased",
                    device=0 if torch.cuda.is_available() else -1
                )
            self.logger.info("Pipeline NER inicializado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar pipeline NER: {str(e)}")
            raise
        
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
        self.vectorizer.fit_transform(self.vantagens_chave)
        
        # Verificar disponibilidade dos modelos
        try:
            # Verificar Llava 13b
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2-vision:11b-instruct-q8_0",
                    "prompt": "test",
                    "stream": False
                }
            )
            if response.status_code == 404:
                self.logger.error("Modelo llama3.2-vision:11b-instruct-q8_0 não encontrado. Por favor, instale usando 'ollama pull llama3.2-vision:11b-instruct-q8_0'")
                raise Exception("Modelo llama3.2-vision:11b-instruct-q8_0 não disponível")
            
            # Verificar Gemma 12b
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "gemma3:12b",
                    "prompt": "test",
                    "stream": False
                }
            )
            if response.status_code == 404:
                self.logger.error("Modelo gemma3:12b não encontrado. Por favor, instale usando 'ollama pull gemma3:12b'")
                raise Exception("Modelo gemma3:12b não disponível")
            
            self.logger.info("Modelos llama3.2:13b e gemma3:12b disponíveis e prontos para uso")
        except Exception as e:
            self.logger.error(f"Erro ao verificar modelos: {str(e)}")
            raise
        
        self.logger.info("Inicialização do PDFProcessor concluída com sucesso")

    def process_pdf(self, pdf_path: str) -> dict:
        try:
            # Processar com Llama Vision
            resultado_llama = self._processar_com_llama_vision(pdf_path)
            if resultado_llama and "erro" not in resultado_llama.lower():
                # Se Llama retornou algo útil, usa o resultado
                texto_completo = self._extrair_texto_digital(pdf_path)
                if len(texto_completo.strip()) < 100:
                    texto_completo = self._processar_como_imagem(pdf_path)
                return {"response": resultado_llama, "texto_extraido": texto_completo}
            
            # Se Llama falhou, tenta com Gemma usando o texto
            texto_completo = self._extrair_texto_digital(pdf_path)
            if len(texto_completo.strip()) < 100:
                texto_completo = self._processar_como_imagem(pdf_path)
            
            resultado_gemma = self._processar_com_gemma(texto_completo)
            return {"response": resultado_gemma, "texto_extraido": texto_completo}
                
        except Exception as e:
            self.logger.error(f"Erro ao processar PDF: {str(e)}")
            return {"response": "", "texto_extraido": ""}

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
                        self.logger.warning(f"Item sem código ou descrição: {item}")
                        continue
                    
                    # Converter valor para float
                    valor = 0.0
                    try:
                        # Limpar o valor antes de converter
                        valor_str = valor_str.replace('R$', '').replace(' ', '').strip()
                        valor = float(valor_str.replace('.', '').replace(',', '.'))
                    except ValueError:
                        self.logger.warning(f"Erro ao converter valor: {valor_str}")
                        continue
                    
                    # Criar objeto Vantagem
                    vantagem = Vantagem(
                        codigo=codigo,
                        descricao=descricao,
                        percentual_duracao=percentual,
                        valor=valor
                    )
                    vantagens.append(vantagem)
                    self.logger.info(f"Vantagem extraída: {vantagem}")
                except Exception as e:
                    self.logger.error(f"Erro ao processar item: {item}")
                    self.logger.error(f"Erro: {str(e)}")
                    continue
                    
        return vantagens

    def _analisar_com_llava(self, img_path: str) -> str:
        try:
            # Converter PDF para imagem
            images = convert_from_path(img_path)
            if not images:
                self.logger.error("Nenhuma imagem extraída do PDF")
                return ""
            
            # Pegar primeira página
            image = images[0]
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Salvar temporariamente como PNG
            temp_img_path = "temp_image.png"
            image.save(temp_img_path, format='PNG')
            
            # Ler a imagem processada
            with open(temp_img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Limpar arquivo temporário
            try:
                os.remove(temp_img_path)
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Erro ao processar imagem: {str(e)}")
            return ""

        # Preparar prompt para Llava
        prompt = """
        Analise esta imagem de um contracheque e extraia as informações no seguinte formato:

        DADOS PESSOAIS:
        Nome: [nome completo]
        Matrícula: [número]
        Mês/Ano Referência: [mês/ano]

        VANTAGENS:
        1. Código: [código]
           Descrição: [descrição]
           Percentual/Horas: [valor se houver]
           Valor: R$ [valor]
        2. Código: [próximo código]
           ...

        TOTAL DE VANTAGENS: R$ [valor total]

        Por favor, mantenha este formato exato, preenchendo os campos entre colchetes com os valores encontrados.
        Mantenha os zeros à esquerda nos códigos.
        Preserve os valores monetários exatamente como aparecem.
        Se algum campo não for encontrado, mantenha o campo mas deixe vazio.
        """
        
        # Enviar para Llava
        try:
            self.logger.info("Enviando requisição para Llava...")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llava:13b",
                    "prompt": prompt,
                    "stream": False,
                    "images": [img_data]
                }
            )
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao usar Llava. Status: {response.status_code}")
                self.logger.error(f"Resposta de erro: {response.text}")
                return ""

            # Retornar resposta bruta
            return response.json().get("response", "")
                
        except Exception as e:
            self.logger.error(f"Erro ao usar Llava: {str(e)}")
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

    def _usar_ollama_fallback(self, texto: str) -> str:
        prompt = f"""
        Você é um especialista em extrair informações de contracheques. Analise o texto abaixo que foi extraído de um contracheque e identifique as informações solicitadas.

        Regras importantes:
        1. Procure por padrões como "NOME:", "MATRÍCULA:", etc.
        2. Na seção de vantagens, procure por códigos numéricos seguidos de descrições e valores
        3. Valores monetários geralmente aparecem no formato "R$ X.XXX,XX" ou apenas números com vírgula
        4. Mantenha os zeros à esquerda nos códigos
        5. Se não encontrar alguma informação, deixe em branco
        6. Não invente ou suponha informações

        Por favor, extraia e organize as informações no seguinte formato:

        DADOS PESSOAIS:
        Nome: [extrair nome completo do servidor]
        Matrícula: [extrair número da matrícula]
        Mês/Ano Referência: [extrair mês e ano]

        VANTAGENS:
        [Liste cada vantagem encontrada no formato:]
        1. Código: [código numérico]
           Descrição: [nome/descrição da vantagem]
           Valor: R$ [valor monetário]

        TOTAL DE VANTAGENS: R$ [valor total das vantagens]

        Texto do contracheque:
        {texto}
        """

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "gemma3:12b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048
                    }
                }
            )

            if response.status_code != 200:
                self.logger.error(f"Erro ao processar texto com Ollama: {response.status_code}")
                return ""

            return response.json().get("response", "")
            
        except Exception as e:
            self.logger.error(f"Erro ao processar texto com Ollama: {str(e)}")
            return ""

    def _normalizar_valor(self, valor_str: str) -> float:
        try:
            valor_str = valor_str.replace("R$", "").strip()
            valor_str = valor_str.replace(".", "").replace(",", ".")
            return float(valor_str)
        except:
            return 0.0

    def _processar_com_llama_vision(self, pdf_path: str) -> str:
        try:
            # Converter PDF para imagem
            images = convert_from_path(pdf_path)
            if not images:
                self.logger.error("Nenhuma imagem extraída do PDF")
                return ""
            
            # Pegar primeira página
            image = images[0]
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Salvar temporariamente como PNG
            temp_img_path = "temp_image.png"
            image.save(temp_img_path, format='PNG')
            
            # Ler a imagem processada
            with open(temp_img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Limpar arquivo temporário
            try:
                os.remove(temp_img_path)
            except:
                pass

            prompt = """
            Analise esta tabela de VANTAGENS de um contracheque. A tabela tem exatamente estas colunas em ordem:

            | Cód. | Descrição | Perct./Horas | Período | Valor(R$) |

            Regras de extração:
            1. Código: Sempre existe e mantém zeros à esquerda (ex: 0022P)
            2. Descrição: Sempre existe (ex: Vencimento Inc)
            3. Perct./Horas: Pode estar vazio ou ter um valor (ex: 30.00)
            4. Período: Pode estar vazio ou ter um valor (ex: 02.2021)
            5. Valor(R$): SEMPRE tem um valor monetário (ex: 2.693,71)

            Extraia cada linha da tabela exatamente neste formato:

            VANTAGENS:
            1. Código: 0022P
               Descrição: Vencimento Inc
               Percentual/Horas: 30.00
               Período: 02.2021
               Valor: R$ 2.693,71

            2. Código: [próximo código]
               ...

            TOTAL DE VANTAGENS: R$ [soma exata de todos os valores]

            IMPORTANTE:
            - A coluna Valor(R$) é a ÚLTIMA coluna da tabela
            - Todo valor em R$ DEVE ter centavos (vírgula e 2 dígitos)
            - Mantenha EXATAMENTE a formatação dos números como está na imagem
            - Não pule nenhuma linha da tabela
            - Não invente valores
            - Se Perct./Horas ou Período estiverem vazios, deixe em branco (não use 0,00)
            - O valor em R$ NUNCA está vazio e SEMPRE deve ser extraído
            """

            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2-vision:11b-instruct-q8_0",
                    "prompt": prompt,
                    "stream": False,
                    "images": [img_data],
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4096,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                }
            )

            if response.status_code != 200:
                self.logger.error(f"Erro ao processar com Llama Vision: {response.status_code}")
                self.logger.error(f"Resposta de erro: {response.text}")
                return ""

            resultado = response.json().get("response", "")
            self.logger.info(f"Resposta do Llama Vision: {resultado}")
            return resultado

        except Exception as e:
            self.logger.error(f"Erro ao processar com Llama Vision: {str(e)}")
            return ""

    def _processar_com_gemma(self, texto: str) -> str:
        prompt = f"""
        Você é um especialista em extrair informações de contracheques. Analise este texto extraído de um contracheque e identifique as informações solicitadas.

        Regras importantes:
        1. Procure por padrões como "NOME:", "MATRÍCULA:", etc.
        2. Na seção de vantagens, procure por códigos numéricos seguidos de descrições e valores
        3. Valores monetários geralmente aparecem no formato "R$ X.XXX,XX" ou apenas números com vírgula
        4. Mantenha os zeros à esquerda nos códigos
        5. Se não encontrar alguma informação, deixe em branco
        6. Não invente ou suponha informações
        7. Preste muita atenção aos detalhes e números

        Por favor, extraia e organize as informações no seguinte formato:

        DADOS PESSOAIS:
        Nome: [nome completo do servidor]
        Matrícula: [número da matrícula]
        Mês/Ano Referência: [mês/ano]

        VANTAGENS:
        [Liste cada vantagem encontrada no formato:]
        1. Código: [código numérico]
           Descrição: [nome/descrição da vantagem]
           Valor: R$ [valor monetário]

        TOTAL DE VANTAGENS: R$ [valor total]

        Texto do contracheque:
        {texto}
        """

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "gemma3:12b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 4096,
                        "top_k": 50,
                        "top_p": 0.9,
                        "context_window": 8192,
                        "stop": ["</s>", "Human:", "Assistant:"]
                    }
                }
            )

            if response.status_code != 200:
                self.logger.error(f"Erro ao processar com Gemma: {response.status_code}")
                self.logger.error(f"Resposta de erro: {response.text}")
                return ""

            resultado = response.json().get("response", "")
            self.logger.info(f"Resposta do Gemma: {resultado}")
            return resultado
            
        except Exception as e:
            self.logger.error(f"Erro ao processar com Gemma: {str(e)}")
            return "" 