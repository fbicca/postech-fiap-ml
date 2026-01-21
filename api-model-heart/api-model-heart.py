# api.py - FastAPI para predição de risco cardíaco (11 inputs, PT/EN, fallback de colunas via X_train.csv)
# Execução: uvicorn api:app --host 0.0.0.0 --port 8001

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from openai import OpenAI

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_insuficiencia_cardiaca.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_dados.pkl")
# Fallback de colunas do treino (usa cabeçalho do CSV para recuperar ordem/nomes)
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "X_train.csv")
# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Modelo barato mas eficiente

app = FastAPI(title="Heart Failure Predictor API", version="1.2.0")

# Inicializar cliente OpenAI (se API key estiver disponível)
openai_client = None
print(f"[DEBUG OpenAI Init] Verificando configuração da OpenAI...")
print(f"[DEBUG OpenAI Init] OPENAI_API_KEY definida: {bool(OPENAI_API_KEY)}")
if OPENAI_API_KEY:
    print(f"[DEBUG OpenAI Init] OPENAI_API_KEY encontrada (primeiros 10 caracteres: {OPENAI_API_KEY[:10]}...)")
    print(f"[DEBUG OpenAI Init] OPENAI_MODEL: {OPENAI_MODEL}")
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"[DEBUG OpenAI Init] ✅ Cliente OpenAI inicializado com sucesso!")
    except Exception as e:
        print(f"[DEBUG OpenAI Init] ❌ Warning: Não foi possível inicializar cliente OpenAI: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[DEBUG OpenAI Init] ⚠️ OPENAI_API_KEY não configurada. Explicações da OpenAI não estarão disponíveis.")


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class Patient(BaseModel):
    # 11 entradas (com normalização PT/EN via validadores)
    Age: int = Field(..., ge=0, le=120)
    Sex: str  # M/F ou masculino/feminino
    ChestPainType: str  # TA/ATA/NAP/ASY + sinônimos
    RestingBP: float  # 70–250
    Cholesterol: float  # 100–600
    FastingBS: int | str | bool  # sim/não, 1/0
    RestingECG: str  # Normal/ST/LVH + sinônimos
    MaxHR: int  # 40–250
    # Aceita Exang e ExerciseAngina; será normalizado para ExerciseAngina ('Y'/'N')
    ExerciseAngina: Optional[str] = None
    Exang: Optional[int | str | bool] = None
    Oldpeak: float | str  # aceita vírgula
    ST_Slope: str  # Up/Flat/Down + sinônimos

    # ---- Validadores (Pydantic v2) ----
    @field_validator('Sex')
    @classmethod
    def norm_sex(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s in {'m','masc','masculino','male','homem'}: return 'M'
        if s in {'f','fem','feminino','female','mulher'}: return 'F'
        raise ValueError("Sexo inválido. Use M/F ou masculino/feminino.")

    @field_validator('ChestPainType')
    @classmethod
    def norm_cpt(cls, v: str) -> str:
        s = str(v).strip().lower()
        mapping = {
            'ta':'TA','típica':'TA','tipica':'TA','typical angina':'TA',
            'ata':'ATA','atípica':'ATA','atipica':'ATA','atypical angina':'ATA',
            'nap':'NAP','não anginosa':'NAP','nao anginosa':'NAP','non-anginal pain':'NAP',
            'asy':'ASY','assintomática':'ASY','assintomatica':'ASY','asymptomatic':'ASY'
        }
        return mapping.get(s, s.upper())

    @field_validator('RestingBP')
    @classmethod
    def check_bp(cls, v) -> float:
        try:
            bp = float(str(v).replace(',', '.'))
        except Exception:
            raise ValueError("RestingBP inválido.")
        if not (70 <= bp <= 250):
            raise ValueError("RestingBP fora do intervalo recomendado (70–250 mmHg).")
        return bp

    @field_validator('Cholesterol')
    @classmethod
    def check_chol(cls, v) -> float:
        try:
            c = float(str(v).replace(',', '.'))
        except Exception:
            raise ValueError("Cholesterol inválido.")
        if not (100 <= c <= 600):
            raise ValueError("Cholesterol fora do intervalo recomendado (100–600 mg/dL).")
        return c

    @field_validator('FastingBS')
    @classmethod
    def norm_fbs(cls, v) -> int:
        s = str(v).strip().lower()
        if s in {'1','true','sim','yes'}: return 1
        if s in {'0','false','nao','não','no'}: return 0
        try:
            n = int(float(s))
            return 1 if n >= 1 else 0
        except Exception:
            raise ValueError("FastingBS inválido (use 1/0, sim/não).")

    @field_validator('RestingECG')
    @classmethod
    def norm_ecg(cls, v: str) -> str:
        s = str(v).strip().lower()
        mapping = {
            'normal':'Normal',
            'st':'ST','st-t wave abnormality':'ST','anormalidade st-t':'ST',
            'lvh':'LVH','left ventricular hypertrophy':'LVH','hipertrofia ventricular esquerda':'LVH'
        }
        return mapping.get(s, s.capitalize())

    @field_validator('MaxHR')
    @classmethod
    def check_hr(cls, v) -> int:
        try:
            hr = int(float(str(v).replace(',', '.')))
        except Exception:
            raise ValueError("MaxHR inválido.")
        if not (40 <= hr <= 250):
            raise ValueError("MaxHR fora do intervalo recomendado (40–250 bpm).")
        return hr

    @field_validator('Oldpeak')
    @classmethod
    def norm_oldpeak(cls, v) -> float:
        try:
            op = float(str(v).replace(',', '.'))
        except Exception:
            raise ValueError("Oldpeak inválido (use número, aceita vírgula).")
        if not (0.0 <= op <= 10.0):
            raise ValueError("Oldpeak fora do intervalo (0.0–10.0).")
        return op

    @field_validator('ST_Slope')
    @classmethod
    def norm_slope(cls, v: str) -> str:
        s = str(v).strip().lower()
        mapping = {
            'up':'Up','ascendente':'Up','asc':'Up',
            'flat':'Flat','plano':'Flat',
            'down':'Down','descendente':'Down','desc':'Down'
        }
        return mapping.get(s, s.capitalize())

    @field_validator('ExerciseAngina', mode='before')
    @classmethod
    def norm_exang1(cls, v):
        if v is None: return None
        s = str(v).strip().lower()
        if s in {'y','yes','sim'}: return 'Y'
        if s in {'n','no','nao','não'}: return 'N'
        return s.upper()

    @field_validator('Exang', mode='before')
    @classmethod
    def norm_exang2(cls, v):
        if v is None: return None
        s = str(v).strip().lower()
        if s in {'1','true','yes','sim'}: return 1
        if s in {'0','false','no','nao','não'}: return 0
        try:
            return 1 if int(float(s))>=1 else 0
        except Exception:
            raise ValueError("Exang inválido (use 1/0, sim/não).")

    @model_validator(mode='after')
    def combine_exang(self):
        # Se Exang foi informado e ExerciseAngina não, convertê-lo (1->'Y', 0->'N')
        if self.Exang is not None and self.ExerciseAngina is None:
            self.ExerciseAngina = 'Y' if int(self.Exang)==1 else 'N'
        # Se ainda ausente, default conservador 'N'
        if self.ExerciseAngina is None:
            self.ExerciseAngina = 'N'
        return self

class PredictResponse(BaseModel):
    prediction: Literal[0,1]
    label: Literal["BAIXO_RISCO","ALTO_RISCO"]
    probability_positive: float
    modelDetails: Dict[str, Any]
    warnings: List[str] = []
    explanation: Optional[str] = None  # Explicação em linguagem natural gerada pela OpenAI (para paciente)
    explanation_patient: Optional[str] = None  # Explicação para o paciente
    explanation_professional: Optional[str] = None  # Explicação técnica para o profissional médico


# ------------------------------------------------------------------------------
# Carregar artefatos
# ------------------------------------------------------------------------------
def _load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError("Modelo e/ou scaler não encontrados. Treine e salve os arquivos .pkl.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

MODEL, SCALER = _load_artifacts()


# ------------------------------------------------------------------------------
# Colunas esperadas (robusto com fallback para CSV)
# ------------------------------------------------------------------------------
def get_expected_columns() -> List[str]:
    """
    Retorna a lista/ordem de colunas esperadas pelo modelo.
    1) Se o modelo tiver feature_names_in_ com nomes de colunas, usa.
    2) Caso contrário, carrega do cabeçalho do X_train.csv (FEATURE_COLUMNS_PATH).
    """
    names = getattr(MODEL, "feature_names_in_", None)
    if names is not None:
        are_strings = all(isinstance(c, (str, bytes)) for c in names)
        if are_strings:
            return list(names)

    # Fallback: cabeçalho do arquivo de treino
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise RuntimeError(
            "Não foi possível determinar as colunas esperadas. "
            "Defina FEATURE_COLUMNS_PATH para um CSV com o cabeçalho correto (ex.: X_train.csv)."
        )
    header = pd.read_csv(FEATURE_COLUMNS_PATH, nrows=0)
    cols = list(header.columns)
    if len(cols) == 0:
        raise RuntimeError(f"O arquivo {FEATURE_COLUMNS_PATH} não possui cabeçalho de colunas.")
    return cols


# ------------------------------------------------------------------------------
# Pré-processamento de uma linha e escala
# ------------------------------------------------------------------------------
def encode_align_scale(df_row: pd.DataFrame):
    """
    Normaliza entradas, faz get_dummies(drop_first=True), alinha para as colunas do treino
    e aplica o scaler, preservando nomes de colunas para evitar warnings do scikit-learn.
    """
    # Se houver Exang mas não ExerciseAngina, inferir
    if 'Exang' in df_row.columns and 'ExerciseAngina' not in df_row.columns:
        df_row = df_row.copy()
        df_row['ExerciseAngina'] = df_row['Exang'].apply(lambda x: 'Y' if int(x)==1 else 'N')

    expected_cols = get_expected_columns()

    # One-Hot consistente com o treino (drop_first=True)
    dummies = pd.get_dummies(df_row, drop_first=True)

    # Adiciona colunas faltantes
    for col in expected_cols:
        if col not in dummies.columns:
            dummies[col] = 0

    # Remove extras e reordena
    dummies = dummies[expected_cols]

    # Checagem opcional de consistência com o scaler
    n_expected = len(expected_cols)
    n_scaler = getattr(SCALER, "n_features_in_", None)
    if n_scaler is not None and n_scaler != n_expected:
        raise RuntimeError(
            f"Incompatibilidade de features: scaler espera {n_scaler} colunas, "
            f"mas o alinhamento gerou {n_expected}. Verifique FEATURE_COLUMNS_PATH."
        )

    # Escala
    scaled = SCALER.transform(dummies)
    # garantir DataFrame com nomes após o scaler
    try:
        import numpy as _np
        if isinstance(scaled, _np.ndarray):
            scaled = pd.DataFrame(scaled, columns=expected_cols, index=dummies.index)
    except Exception:
        pass
    return scaled, expected_cols


# ------------------------------------------------------------------------------
# Geração de explicações com OpenAI
# ------------------------------------------------------------------------------
def generate_explanation(patient: Patient, prediction: int, label: str, probability: float) -> Optional[Dict[str, Optional[str]]]:
    """
    Gera explicações em linguagem natural do diagnóstico usando OpenAI.
    Retorna um dict com duas explicações:
    - 'patient': explicação simples e acessível para o paciente
    - 'professional': explicação técnica detalhada para o profissional médico
    - 'full': resposta completa (para compatibilidade)
    Retorna None se a API não estiver configurada ou houver erro.
    """
    print(f"[DEBUG OpenAI] Iniciando geração de explicação...")
    print(f"[DEBUG OpenAI] Cliente OpenAI inicializado: {openai_client is not None}")
    print(f"[DEBUG OpenAI] OPENAI_API_KEY configurada: {bool(OPENAI_API_KEY)}")
    print(f"[DEBUG OpenAI] Modelo configurado: {OPENAI_MODEL}")
    
    if not openai_client:
        print("[DEBUG OpenAI] ❌ Cliente OpenAI não está inicializado. Retornando None.")
        return None
    
    try:
        print(f"[DEBUG OpenAI] ✅ Cliente OpenAI disponível. Gerando explicação...")
        # Mapear valores para descrições mais legíveis
        sex_map = {"M": "masculino", "F": "feminino"}
        chest_pain_map = {
            "TA": "angina típica",
            "ATA": "angina atípica",
            "NAP": "dor não anginosa",
            "ASY": "assintomática"
        }
        ecg_map = {
            "Normal": "normal",
            "ST": "anormalidade da onda ST-T",
            "LVH": "hipertrofia ventricular esquerda"
        }
        slope_map = {
            "Up": "ascendente",
            "Flat": "plano",
            "Down": "descendente"
        }
        
        # Preparar dados do paciente em formato legível
        patient_data = f"""
Dados do paciente:
- Idade: {patient.Age} anos
- Sexo: {sex_map.get(patient.Sex, patient.Sex)}
- Tipo de dor no peito: {chest_pain_map.get(patient.ChestPainType, patient.ChestPainType)}
- Pressão arterial em repouso: {patient.RestingBP} mmHg
- Colesterol: {patient.Cholesterol} mg/dL
- Glicemia de jejum elevada: {"Sim" if patient.FastingBS == 1 else "Não"}
- ECG em repouso: {ecg_map.get(patient.RestingECG, patient.RestingECG)}
- Frequência cardíaca máxima: {patient.MaxHR} bpm
- Angina induzida por exercício: {"Sim" if patient.ExerciseAngina == "Y" else "Não"}
- Depressão do segmento ST (Oldpeak): {patient.Oldpeak}
- Inclinação do segmento ST: {slope_map.get(patient.ST_Slope, patient.ST_Slope)}
"""
        
        # Preparar prompt
        risk_level = "ALTO RISCO" if prediction == 1 else "BAIXO RISCO"
        probability_percent = f"{probability * 100:.1f}%"
        
        prompt = f"""Você é um assistente médico especializado em cardiologia. Analise os dados do paciente e o resultado do modelo de predição de risco cardíaco e forneça DUAS explicações distintas em português brasileiro.

{patient_data}

Resultado da predição:
- Nível de risco: {risk_level}
- Probabilidade de doença cardiovascular: {probability_percent}

Por favor, forneça DUAS explicações separadas:

## 1. EXPLICAÇÃO PARA O PACIENTE
Forneça uma explicação em linguagem natural, simples e acessível que:
- Explique o que significa o resultado do diagnóstico de forma clara e compreensível
- Use linguagem simples, evitando termos técnicos complexos
- Destaque os principais fatores de risco identificados de forma educativa
- Mantenha um tom empático, acolhedor e tranquilizador
- Inclua orientações práticas sobre próximos passos
- Seja concisa (máximo 150 palavras)
- IMPORTANTE: Sempre enfatize que esta é apenas uma predição baseada em modelo de machine learning e que é essencial consultar um médico para diagnóstico definitivo

## 2. EXPLICAÇÃO PARA O PROFISSIONAL MÉDICO
Forneça uma explicação técnica e detalhada que:
- Analise os dados clínicos do paciente de forma técnica
- Destaque os fatores de risco específicos identificados e sua relevância clínica
- Explique a correlação entre os parâmetros e o resultado da predição
- Inclua considerações sobre a confiabilidade do modelo e limitações
- Sugira possíveis investigações complementares se necessário
- Use terminologia médica apropriada
- Seja concisa mas completa (máximo 200 palavras)
- Inclua observações sobre a interpretação clínica do resultado

FORMATO DE RESPOSTA:
Use o seguinte formato exato, separando as duas explicações:

---EXPLICAÇÃO_PACIENTE---
[sua explicação para o paciente aqui]
---FIM_EXPLICAÇÃO_PACIENTE---

---EXPLICAÇÃO_PROFISSIONAL---
[sua explicação técnica para o profissional aqui]
---FIM_EXPLICAÇÃO_PROFISSIONAL---"""

        print(f"[DEBUG OpenAI] Prompt preparado. Tamanho: {len(prompt)} caracteres")
        print(f"[DEBUG OpenAI] Dados do paciente: Idade={patient.Age}, Sexo={patient.Sex}, Risco={risk_level}, Prob={probability_percent}")
        
        # Chamar API da OpenAI
        print(f"[DEBUG OpenAI] 📤 Enviando requisição para OpenAI (modelo: {OPENAI_MODEL})...")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente médico especializado em cardiologia que explica resultados de exames e predições de risco cardíaco de forma clara e compreensível."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800  # Aumentado para acomodar duas explicações
        )
        
        print(f"[DEBUG OpenAI] ✅ Resposta recebida da OpenAI")
        print(f"[DEBUG OpenAI] ID da resposta: {response.id}")
        print(f"[DEBUG OpenAI] Modelo usado: {response.model}")
        print(f"[DEBUG OpenAI] Tokens usados: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
        print(f"[DEBUG OpenAI] Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
        
        full_response = response.choices[0].message.content.strip()
        print(f"[DEBUG OpenAI] ✅ Resposta completa gerada com sucesso!")
        print(f"[DEBUG OpenAI] Tamanho da resposta completa: {len(full_response)} caracteres")
        
        # Processar e separar as duas explicações
        explanation_patient = None
        explanation_professional = None
        
        try:
            # Extrair explicação do paciente
            if "---EXPLICAÇÃO_PACIENTE---" in full_response and "---FIM_EXPLICAÇÃO_PACIENTE---" in full_response:
                start_patient = full_response.find("---EXPLICAÇÃO_PACIENTE---") + len("---EXPLICAÇÃO_PACIENTE---")
                end_patient = full_response.find("---FIM_EXPLICAÇÃO_PACIENTE---")
                explanation_patient = full_response[start_patient:end_patient].strip()
            elif "EXPLICAÇÃO PARA O PACIENTE" in full_response or "EXPLICAÇÃO_PACIENTE" in full_response:
                # Fallback: tentar extrair sem marcadores exatos
                parts = full_response.split("EXPLICAÇÃO PARA O PACIENTE")
                if len(parts) > 1:
                    explanation_patient = parts[1].split("EXPLICAÇÃO PARA O PROFISSIONAL")[0].strip()
            
            # Extrair explicação profissional
            if "---EXPLICAÇÃO_PROFISSIONAL---" in full_response and "---FIM_EXPLICAÇÃO_PROFISSIONAL---" in full_response:
                start_prof = full_response.find("---EXPLICAÇÃO_PROFISSIONAL---") + len("---EXPLICAÇÃO_PROFISSIONAL---")
                end_prof = full_response.find("---FIM_EXPLICAÇÃO_PROFISSIONAL---")
                explanation_professional = full_response[start_prof:end_prof].strip()
            elif "EXPLICAÇÃO PARA O PROFISSIONAL" in full_response or "EXPLICAÇÃO_PROFISSIONAL" in full_response:
                # Fallback: tentar extrair sem marcadores exatos
                parts = full_response.split("EXPLICAÇÃO PARA O PROFISSIONAL")
                if len(parts) > 1:
                    explanation_professional = parts[1].strip()
            
            # Se não conseguiu separar, usar a resposta completa como explicação do paciente (compatibilidade)
            if not explanation_patient and not explanation_professional:
                explanation_patient = full_response
                print(f"[DEBUG OpenAI] ⚠️ Não foi possível separar as explicações. Usando resposta completa como explicação do paciente.")
            else:
                print(f"[DEBUG OpenAI] ✅ Explicações separadas com sucesso!")
                print(f"[DEBUG OpenAI] Tamanho explicação paciente: {len(explanation_patient) if explanation_patient else 0} caracteres")
                print(f"[DEBUG OpenAI] Tamanho explicação profissional: {len(explanation_professional) if explanation_professional else 0} caracteres")
        
        except Exception as parse_error:
            print(f"[DEBUG OpenAI] ⚠️ Erro ao processar explicações: {parse_error}")
            # Em caso de erro no parsing, usar a resposta completa como explicação do paciente
            explanation_patient = full_response
        
        # Retornar dict com ambas as explicações
        return {
            "patient": explanation_patient,
            "professional": explanation_professional,
            "full": full_response  # Manter compatibilidade
        }
        
    except Exception as e:
        # Em caso de erro, não quebrar a API, apenas não retornar explicação
        print(f"[DEBUG OpenAI] ❌ ERRO ao gerar explicação com OpenAI:")
        print(f"[DEBUG OpenAI] Tipo do erro: {type(e).__name__}")
        print(f"[DEBUG OpenAI] Mensagem do erro: {str(e)}")
        import traceback
        print(f"[DEBUG OpenAI] Traceback completo:")
        traceback.print_exc()
        return None


# ------------------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "scaler_loaded": os.path.exists(SCALER_PATH),
        "feature_columns_source": (
            "model.feature_names_in_" if getattr(MODEL, "feature_names_in_", None) is not None else FEATURE_COLUMNS_PATH
        )
    }


@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient):
    warnings = []
    try:
        df = pd.DataFrame([patient.dict()])
        X_scaled_df, cols = encode_align_scale(df)

        if hasattr(MODEL, "predict_proba"):
            proba = float(MODEL.predict_proba(X_scaled_df)[:, 1][0])
        else:
            raw = MODEL.decision_function(X_scaled_df)[0]
            proba = float(1 / (1 + np.exp(-raw)))

        pred = int(MODEL.predict(X_scaled_df)[0])
        label = "ALTO_RISCO" if pred == 1 else "BAIXO_RISCO"

        # Gerar explicação usando OpenAI
        print(f"[DEBUG /predict] Gerando explicação para predição: {pred}, label: {label}, prob: {proba:.4f}")
        explanation_result = generate_explanation(patient, pred, label, proba)
        print(f"[DEBUG /predict] Resultado da explicação recebido: {type(explanation_result)}")
        
        # Processar resultado da explicação
        explanation_patient = None
        explanation_professional = None
        explanation_full = None
        
        if explanation_result:
            if isinstance(explanation_result, dict):
                explanation_patient = explanation_result.get("patient")
                explanation_professional = explanation_result.get("professional")
                explanation_full = explanation_result.get("full") or explanation_patient
            elif isinstance(explanation_result, str):
                # Compatibilidade: se retornar string, usar como explicação do paciente
                explanation_patient = explanation_result
                explanation_full = explanation_result
        
        print(f"[DEBUG /predict] Explicação paciente: {bool(explanation_patient)}")
        print(f"[DEBUG /predict] Explicação profissional: {bool(explanation_professional)}")
        
        # Manter 'explanation' para compatibilidade com código existente (usa explicação do paciente)
        return {
            "prediction": pred,
            "label": label,
            "probability_positive": proba,
            "modelDetails": {
                "features_expected": cols,
                "model_class": type(MODEL).__name__,
            },
            "warnings": warnings,
            "explanation": explanation_full,  # Compatibilidade: explicação do paciente
            "explanation_patient": explanation_patient,
            "explanation_professional": explanation_professional,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class BatchRequest(BaseModel):
    items: List[Patient]


@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    try:
        df = pd.DataFrame([p.dict() for p in payload.items])
        X_scaled_df, _ = encode_align_scale(df)
        preds = MODEL.predict(X_scaled_df).astype(int).tolist()
        if hasattr(MODEL, "predict_proba"):
            probas = MODEL.predict_proba(X_scaled_df)[:, 1].astype(float).tolist()
        else:
            raw = MODEL.decision_function(X_scaled_df)
            probas = (1 / (1 + np.exp(-raw))).astype(float).tolist()
        labels = ["ALTO_RISCO" if p == 1 else "BAIXO_RISCO" for p in preds]
        return {"predictions": preds, "labels": labels, "probabilities_positive": probas}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------------------
# Endpoint de debug para inspecionar o vetor alinhado/escalado
# ------------------------------------------------------------------------------
@app.post("/debug-vector")
def debug_vector(patient: Patient):
    try:
        df = pd.DataFrame([patient.dict()])
        X_scaled_df, cols = encode_align_scale(df)
        # Retorna apenas uma amostra (primeiros 12 valores) para não poluir
        sample = X_scaled_df[0][:min(12, X_scaled_df.shape[1])].tolist()
        return {
            "n_features": len(cols),
            "cols_sample": cols[:min(12, len(cols))],
            "vector_sample": sample,
            "feature_columns_source": (
                "model.feature_names_in_" if getattr(MODEL, "feature_names_in_", None) is not None else FEATURE_COLUMNS_PATH
            )
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
